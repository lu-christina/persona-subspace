#!/usr/bin/env python
import pathlib, h5py, numpy as np, html, re
from flask import Flask, Response
from transformers import AutoTokenizer

# ---------- CONFIG -------------------------------------------------------
MODEL_NAME   = "meta-llama/Meta-Llama-3.1-8B-Instruct"

CHAT_H5      = pathlib.Path("/workspace/sae/llama-3-8b-instruct/feature_mining/resid_post_layer_19/trainer_1/chat_topk.h5")
PT_H5        = pathlib.Path("/workspace/sae/llama-3-8b-instruct/feature_mining/resid_post_layer_19/trainer_1/pt_topk.h5")
SIM_H5       = pathlib.Path("/workspace/sae/llama-3-8b-instruct/feature_mining/resid_post_layer_19/trainer_1/embed_unembed_similarity.h5")

TOK_BEFORE = TOK_AFTER = 15
CHARS_LEFT   = 60
PREVIEW_LEN  = 120
TOP_N        = 20
PORT         = 7860
# -------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID    = tokenizer.pad_token_id

h5_chat   = h5py.File(CHAT_H5, "r")
h5_pt     = h5py.File(PT_H5,   "r")
h5_sim    = h5py.File(SIM_H5,  "r")

# ---------- helper functions --------------------------------------------
def rgba(alpha: float) -> str:
    alpha = max(alpha, 0.05)
    return f"rgba(216,68,0,{alpha:.3f})"

SPACE_SYM, NL_SYM, TAB_SYM = "␣", "↵", "⇥"

def show(tok_id: int) -> str:
    tok_id = int(tok_id)
    raw = tokenizer.decode([tok_id], skip_special_tokens=False)
    if raw.startswith("<|") and raw.endswith("|>"):
        return raw
    rep = repr(raw)[1:-1]
    rep = (SPACE_SYM if rep == " " else
           rep.replace("\\n", NL_SYM).replace("\\t", TAB_SYM))
    return html.escape(rep, quote=False)

# scientific‑notation helper
_sci_re = re.compile(r"([0-9.]+)e([+-]?)(\d+)")

def sci_notation(val: float, sig: int = 2) -> str:
    """Return value in a human‑friendly scientific‑notation string, e.g. 2.3x10-3."""
    if val == 0.0:
        return "0"
    s = f"{val:.{sig}e}"
    m = _sci_re.fullmatch(s)
    if not m:
        return s
    coeff, sign, expo = m.groups()
    sign = "-" if sign == "-" else ""  # drop + sign for readability
    return f"{coeff}x10{sign}{int(expo)}"

# ---------- example rendering -------------------------------------------

def example_html(tok_ids: np.ndarray, acts: np.ndarray, score: float) -> str:
    # 1 ── remove *trailing* PADs (keep internal ones)
    while len(tok_ids) > 1 and tok_ids[-2] == PAD_ID:
        tok_ids, acts = tok_ids[:-1], acts[:-1]

    # 2 ── printable pieces and absolute char offsets
    pieces, starts = [], []
    pos = 0
    for tid in tok_ids:
        txt = show(tid)                   # printable string for token
        pieces.append(txt)
        starts.append(pos)
        pos += len(txt)
    ends, total_len = [s + len(p) for s, p in zip(starts, pieces)], pos

    # 3 ── identify peak token
    peak_idx      = int(acts.argmax())
    peak_piece    = pieces[peak_idx]
    peak_start    = starts[peak_idx]
    peak_len      = len(peak_piece)
    peak_end      = peak_start + peak_len

    # 4 ── character window boundaries (do *not* shift left)
    left_chars_avail  = min(peak_start, CHARS_LEFT)
    window_start_char = peak_start - left_chars_avail
    window_end_char   = min(total_len, peak_end + (PREVIEW_LEN - CHARS_LEFT - peak_len))

    raw_window = "".join(pieces)[window_start_char:window_end_char]

    # 5 ── pad to fixed width
    pad_left  = " " * (CHARS_LEFT - left_chars_avail)
    needed_right = PREVIEW_LEN - len(raw_window) - len(pad_left)
    pad_right = " " * max(0, needed_right)
    window = pad_left + raw_window + pad_right

    # 6 ── collect tokens overlapping the window for colouring
    win_tokens, win_acts = [], []
    for piece, act, s, e in zip(pieces, acts, starts, ends):
        if e <= window_start_char or s >= window_end_char:
            continue
        cut_l = max(window_start_char - s, 0)
        cut_r = len(piece) - max(e - window_end_char, 0)
        win_tokens.append(piece[cut_l:cut_r])
        win_acts.append(act)
    win_max = max(win_acts) if win_acts else 1.0

    # 7 ── build coloured spans (peak has its own class)
    spans, cursor = [], len(pad_left)
    for token_piece, act in zip(win_tokens, win_acts):
        is_peak = cursor == CHARS_LEFT    # after left-pad, peak starts here
        cls     = "peak" if is_peak else "preview-tok"
        style   = f"background:{rgba(max(act / win_max, 0.05))};"
        spans.append(
            f'<span class="{cls}" title="{act:.4f}" style="{style}">' \
            f'{html.escape(token_piece, quote=False)}</span>'
        )
        cursor += len(token_piece)

    preview_html = f'<span class="mono-window">{pad_left}{"".join(spans)}{pad_right}</span>' \
                   f'<span class="score">(score {score:.3f})</span>'

    # 8 ── expanded (full) sequence with thin separators
    full_max = acts.max() or 1.0
    full = "".join(
        f'<span class="tok" title="{a:.4f}" '
        f'style="background:{rgba(max(a/full_max,0.05))};">'
        f'{html.escape(p)}</span>'
        for p, a in zip(pieces, acts)
    )

    # 9 ── wrap block
    return (
        '<div class="example collapsed" onclick="this.classList.toggle(\'collapsed\')">'
        f'  <div class="snippet">{preview_html}</div>'
        f'  <div class="full">{full}</div>'
        '</div>'
    )

# ---------- similarity panel --------------------------------------------

def sim_panel(fid: int) -> str:
    F = h5_sim["in_top_tokens"].shape[0]
    if fid < 0 or fid >= F:
        return ""

    def strip(token_ids, css: str):
        pills = "".join(f"<span class='pill'>{show(t)}</span>" for t in token_ids)
        return f"<div class='token-strip {css}'>{pills}</div>"

    enc_top = strip(h5_sim["in_top_tokens"][fid],    "top")
    enc_bot = strip(h5_sim["in_bottom_tokens"][fid], "bottom")
    dec_top = strip(h5_sim["out_top_tokens"][fid],   "top")
    dec_bot = strip(h5_sim["out_bottom_tokens"][fid],"bottom")

    rows = (
        "<table class='sim'>"
        "<tr><th>Embedding → Encoder</th><th>Decoder → Unembedding</th></tr>"
        f"<tr class='top-row'><td>{enc_top}</td><td>{dec_top}</td></tr>"
        f"<tr class='bottom-row'><td>{enc_bot}</td><td>{dec_bot}</td></tr>"
        "</table>"
    )
    return f"<div class='sim-panel'>{rows}</div>"

# ---------- feature block -----------------------------------------------

def feature_block(fid: int) -> str:
    # bounds check on the two mining files (sizes must match)
    F = h5_chat["scores"].shape[0]
    if fid < 0 or fid >= F:
        return f"<p style='color:red'>Feature {fid} out of range.</p>"

    # ----- similarity panel ---------------------------------------------
    html_rows = [f"<h2>Feature {fid}</h2>", sim_panel(fid), "<hr/>"]

    # helper to build a toggle section
    def section(name, h5file, css_id):
        tok   = h5file["tokens"][fid][:TOP_N]
        acts  = h5file["sae_acts"][fid][:TOP_N]
        score = h5file["scores"][fid][:TOP_N]
        freq  = h5file["frequency"][fid]
        total = h5file.attrs["tokens_seen"]
        freq_ratio = freq / total if total else 0.0
        freq_str = sci_notation(freq_ratio)

        examples = "\n".join(example_html(tok[k], acts[k], score[k])
                             for k in range(len(tok)))
        header = (
            f"<div class='sec-header' onclick=\""
            f'document.getElementById(\'{css_id}\').classList.toggle(\'collapsed\')\">'
            f"<b>{name}</b> &nbsp;<span class='freq'>({freq_str})</span>"
            "</div>"
        )
        body = f"<div id='{css_id}' class='sec-body collapsed'>{examples}</div>"
        return header + body

    html_rows.append(section("CHAT",     h5_chat, "chat-sec"))
    html_rows.append(section("PRETRAIN", h5_pt,   "pt-sec"))
    return "\n".join(html_rows)

# ---------- Flask routes -------------------------------------------------
@app.route("/")
def index():
    return Response(f"""
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>SAE Feature Explorer</title>
<style>
 @import url("https://fonts.googleapis.com/css2?family=Noto+Sans+Mono:wght@400&display=swap");
 body {{ font-family:'Noto Sans Mono', monospace; margin:20px; }}

 /* control panel */
 #ctrl {{ margin-bottom:16px; }}
 input[type=number] {{ width:120px; font-size:1em; }}

.sim {{
  table-layout: fixed;
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9em;
}}

.sim th, .sim td {{
  width: 50%;            /* two equal columns */
  border: 1px solid #ccc;
  padding: 2px 4px;
  vertical-align: top;
}}

/* scrolling strip inside each cell */
.token-strip {{
  display: block;
  white-space: nowrap;
  overflow-x: auto;
  padding: 2px;
  border-radius: 4px;
}}

.token-strip.top    {{ background: #e8f9e8; }}
.token-strip.bottom {{ background: #fbeaea; }}

.pill {{
  display: inline-block;
  padding: 1px 4px;
  margin: 1px;
  border: 1px solid #bbb;
  border-radius: 6px;
}}

 /* section headers */
 .sec-header {{ cursor:pointer; background:#eee; padding:6px 8px;
               border:1px solid #ccc; border-radius:4px; margin-top:12px; }}
 .sec-header:hover {{ background:#ddd; }}
 .freq {{ color:#555; font-size:0.85em; }}

 /* collapsible bodies */
 .sec-body {{ border:1px solid #ccc; border-top:none; padding:8px; }}
 .sec-body.collapsed {{ display:none; }}

 /* (original) example card & preview styles - unchanged */
 .example {{ border:1px solid #c3c3c3; margin:14px 0; padding:10px;
             border-radius:6px; background:#fafafa; }}
 .snippet {{ display:flex; white-space:pre; overflow-x:auto; }}
 .mono-window {{ white-space:pre; }}
 .peak {{ background:#d84400; color:#fff; border-radius:8px; padding:0; }}
 .preview-tok {{ padding:0; }}
 .score {{ margin-left:auto; color:#555; font-size:0.9em; }}
 .tok {{ display:inline-block; padding:0 1px;
         border-right:1px solid rgba(0,0,0,0.08); }}
 .tok:last-child {{ border-right:none; }}
 .tok:hover {{ border-right-color:#888; }}
 .full {{ display:none; white-space:pre-wrap; margin-top:12px;
          padding-top:6px; border-top:1px dashed #ccc; }}
 .example.collapsed .full {{ display:none; }}
 .example:not(.collapsed) .full {{ display:block; }}
</style>

</head><body>
<div id="ctrl">
  <label>Feature:
    <input id="fid" type="number" min="0" step="1"
           onkeyup="if(event.key==='Enter') load()">
  </label>
  <button onclick="load()">Load</button>
</div>
<div id="out"></div>
<script>
async function load() {{
  const id = document.getElementById('fid').value;
  if (id === '') return;
  const html = await (await fetch(`/feature/${{id}}`)).text();
  document.getElementById('out').innerHTML = html;
}}
</script>
</body></html>
""", mimetype="text/html")


@app.route("/feature/<int:fid>")
def feature(fid: int):
    return Response(feature_block(fid), mimetype="text/html")

# ---------- run ----------------------------------------------------------
if __name__ == "__main__":
    print(f" * running on http://localhost:{PORT}/")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

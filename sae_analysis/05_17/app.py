#!/usr/bin/env python
import pathlib, h5py, numpy as np, html
from flask import Flask, Response
from transformers import AutoTokenizer

# ---------- CONFIG -------------------------------------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
H5_PATH    = pathlib.Path("feature_mining_runs/run2_hdf5/topk_final.h5")
TOK_BEFORE = TOK_AFTER = 15
CHARS_LEFT   = 60   # how many characters to show *before* the peak token
PREVIEW_LEN  = 120   # total characters in the preview line
TOP_N      = 20
PORT       = 7860
# -------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID    = tokenizer.pad_token_id

h5        = h5py.File(H5_PATH, "r")

# ---------- helper functions --------------------------------------------
def rgba(alpha: float) -> str:
    alpha = max(alpha, 0.05)
    return f"rgba(216,68,0,{alpha:.3f})"

import re, html

SPACE_SYM = "␣"
NL_SYM    = "↵"
TAB_SYM   = "⇥"

def show(tok_id) -> str:
    tok_id = int(tok_id)

    raw = tokenizer.decode([tok_id], skip_special_tokens=False)

    if raw.startswith("<|") and raw.endswith("|>"):
        return raw                      # keep < and >

    rep = repr(raw)[1:-1]               # strip quotes
    rep = (SPACE_SYM if rep == " " else
           rep.replace("\\n", NL_SYM).replace("\\t", TAB_SYM))

    return html.escape(rep, quote=False)   # ‹—— apostrophes stay `'`

TOKENS_LEFT = TOK_BEFORE           # ← keep this constant next to your CONFIG

def example_html(tok_ids: np.ndarray, acts: np.ndarray, score: float) -> str:
    # 1 ── remove *trailing* PADs (keep internal ones)
    while len(tok_ids) and tok_ids[-1] == PAD_ID:
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
            f'<span class="{cls}" title="{act:.4f}" style="{style}">'
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


def feature_block(fid: int) -> str:
    if fid < 0 or fid >= h5["scores"].shape[0]:
        return f"<p style='color:red'>Feature {fid} out of range.</p>"
    tok   = h5["tokens"][fid][:TOP_N]
    acts  = h5["sae_acts"][fid][:TOP_N]
    score = h5["scores"][fid][:TOP_N]
    rows  = [f"<h2>Feature {fid}</h2><hr/>"]
    rows += [example_html(tok[k], acts[k], score[k]) for k in range(len(tok))]
    return "\n".join(rows)

# ---------- Flask routes -------------------------------------------------
@app.route("/")
def index():
    return Response("""
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>SAE Feature Explorer</title>
<style>
 @import url("https://fonts.googleapis.com/css2?family=Noto+Sans+Mono:wght@400&display=swap");
 body { font-family:'Noto Sans Mono', monospace; margin:20px; }

 /* control panel */
 #ctrl { margin-bottom:16px; }
 input[type=number] { width:120px; font-size:1em; }

 /* outer card */
 .example { border:1px solid #c3c3c3; margin:14px 0; padding:10px;
            border-radius:6px; background:#fafafa; }
                    
/* preview container: flex -> score floats right */
.snippet        { display:flex; white-space:pre; overflow-x:auto; }

/* monospaced text area inside the preview */
.mono-window    { white-space:pre; font-family:'Noto Sans Mono', monospace; }

/* highlighted peak token */
.peak { background:#d84400; color:#fff; border-radius:8px; padding:0; }

/* non-peak tokens in the preview - no padding, inline */
.preview-tok { padding:0; }

/* score floats to the far right */
.score          { margin-left:auto; color:#555; font-size:0.9em; }

/* (unchanged) token styling for expanded view */
.tok { display:inline-block; padding:0 1px;
       border-right:1px solid rgba(0,0,0,0.08); }
.tok:last-child { border-right:none; }
.tok:hover      { border-right-color:#888; }

/* full text (shown on click) */
.full { display:none; white-space:pre-wrap; margin-top:12px;
        padding-top:6px; border-top:1px dashed #ccc; }
.example.collapsed .full { display:none; }
.example:not(.collapsed) .full { display:block; }
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
async function load() {
  const id = document.getElementById('fid').value;
  if (id === '') return;
  const html = await (await fetch(`/feature/${id}`)).text();
  document.getElementById('out').innerHTML = html;
}
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

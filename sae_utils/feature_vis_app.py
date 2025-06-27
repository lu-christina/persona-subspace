#!/usr/bin/env python
import pathlib, h5py, html, re, contextlib, json
from flask import Flask, Response, request
from transformers import AutoTokenizer

# ---------- CONFIG -------------------------------------------------------
# MODEL_NAME   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# BASE_DIR     = pathlib.Path("/workspace/sae/llama-3-8b-instruct/feature_mining")
# DEFAULT_LAYER, DEFAULT_TRAINER = 11, 1
MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"
BASE_DIR     = pathlib.Path("/workspace/sae/qwen2.5-7b-instruct/feature_mining")
DEFAULT_LAYER, DEFAULT_TRAINER = 19, 1
CHARS_LEFT, PREVIEW_LEN, TOP_N, PORT = 60, 120, 20, 7863
LABELS_PATH  = BASE_DIR / "feature_labels.json"          # ← NEW
# -------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)

# ---------- label DB -----------------------------------------------------
def _safe_load_labels():
    if not LABELS_PATH.exists():
        LABELS_PATH.write_text("{}")
        return {}
    try:
        with open(LABELS_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        LABELS_PATH.write_text("{}")
        return {}

def _save_labels():
    with open(LABELS_PATH, "w") as f:
        json.dump(LABELS, f, indent=2)

def _lbl_key(layer:int, trainer:int, fid:int) -> str:
    return f"{layer}:{trainer}:{fid}"

LABELS = _safe_load_labels()                              # in-memory cache

# ---------- tokenizer ----------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

# ---------- variant handling --------------------------------------------
current_layer, current_trainer = DEFAULT_LAYER, DEFAULT_TRAINER
def variant_path(layer:int, trainer:int):
    return BASE_DIR / f"resid_post_layer_{layer}" / f"trainer_{trainer}"
def open_h5_group(layer:int, trainer:int):
    root = variant_path(layer, trainer)
    return (h5py.File(root/"chat_topk.h5","r"),
            h5py.File(root/"pt_topk.h5","r"),
            h5py.File(root/"embed_unembed_similarity.h5","r"))
h5_chat, h5_pt, h5_sim = open_h5_group(current_layer, current_trainer)
def switch_variant(layer:int, trainer:int):
    global h5_chat,h5_pt,h5_sim,current_layer,current_trainer
    new_chat,new_pt,new_sim=open_h5_group(layer,trainer)
    with contextlib.suppress(Exception):
        h5_chat.close(),h5_pt.close(),h5_sim.close()
    h5_chat,h5_pt,h5_sim=new_chat,new_pt,new_sim
    current_layer,current_trainer=layer,trainer

# ---------- helpers (unchanged) -----------------------------------------
def rgba(a:float)->str: return f"rgba(216,68,0,{max(a,0.05):.3f})"
SPACE_SYM,NL_SYM,TAB_SYM="␣","↵","⇥"
def show(t:int)->str:
    raw=tokenizer.decode([int(t)],skip_special_tokens=False)
    if raw.startswith("<|") and raw.endswith("|>"): return raw
    rep=repr(raw)[1:-1].replace("\\n",NL_SYM).replace("\\t",TAB_SYM)
    return SPACE_SYM if rep==" " else html.escape(rep,False)
_sci_re=re.compile(r"([0-9.]+)e([+-]?)(\d+)")
def sci(v:float,sig:int=2)->str:
    if v==0: return "0"
    s=f"{v:.{sig}e}";m=_sci_re.fullmatch(s)
    return s if not m else f"{m[1]}x10{'' if m[2]=='+' else '-'}{int(m[3])}"

# ---------- example & similarity panels (unchanged) ---------------------
def example_html(tok_ids,acts,score):
    while len(tok_ids)>1 and tok_ids[-2]==PAD_ID: tok_ids,acts=tok_ids[:-1],acts[:-1]
    pieces,starts,pos=[],[],0
    for tid in tok_ids: txt=show(tid);pieces.append(txt);starts.append(pos);pos+=len(txt)
    ends,total=[s+len(p) for s,p in zip(starts,pieces)],pos
    peak=acts.argmax();ps=starts[peak];pl=len(pieces[peak]);pe=ps+pl
    left=min(ps,CHARS_LEFT);ws, we = ps-left, min(total, pe+(PREVIEW_LEN-CHARS_LEFT-pl))
    raw="".join(pieces)[ws:we]; padl=" "*(CHARS_LEFT-left); padr=" "*(PREVIEW_LEN-len(raw)-len(padl))
    win_tokens,win_acts=[],[]
    for p,a,s,e in zip(pieces,acts,starts,ends):
        if e<=ws or s>=we: continue
        win_tokens.append(p[max(ws-s,0):len(p)-max(e-we,0)]);win_acts.append(a)
    wmax = max(win_acts) if win_acts else 1.0
    cursor = len(padl)
    spans  = []
    for tp, a in zip(win_tokens, win_acts):
        cls = "peak" if cursor == CHARS_LEFT else "preview-tok"
        spans.append(
            f'<span class="{cls}" title="{a:.4f}" '
            f'style="background:{rgba(max(a/wmax,0.05))};">{html.escape(tp,False)}</span>')
        cursor += len(tp)
    preview=f'<span class="mono-window">{padl}{"".join(spans)}{padr}</span><span class="score">(score {score:.3f})</span>'
    fmax=acts.max() or 1.0
    full="".join(f'<span class="tok" title="{a:.4f}" style="background:{rgba(max(a/fmax,0.05))};">{html.escape(p)}</span>' for p,a in zip(pieces,acts))
    return ('<div class="example collapsed" onclick="this.classList.toggle(\'collapsed\')">'
            f'<div class="snippet">{preview}</div><div class="full">{full}</div></div>')
def sim_panel(fid:int)->str:
    F=h5_sim["in_top_tokens"].shape[0]
    if fid<0 or fid>=F: return ""
    def strip(tok,css): return "<div class='token-strip "+css+"'>"+"".join(f"<span class='pill'>{show(t)}</span>" for t in tok)+"</div>"
    enc_t,enc_b=h5_sim["in_top_tokens"][fid],h5_sim["in_bottom_tokens"][fid]
    dec_t,dec_b=h5_sim["out_top_tokens"][fid],h5_sim["out_bottom_tokens"][fid]
    return ("<div class='sim-panel'><table class='sim'>"
            "<tr><th>Embedding → Encoder</th><th>Decoder → Unembedding</th></tr>"
            f"<tr class='top-row'><td>{strip(enc_t,'top')}</td><td>{strip(dec_t,'top')}</td></tr>"
            f"<tr class='bottom-row'><td>{strip(enc_b,'bottom')}</td><td>{strip(dec_b,'bottom')}</td></tr>"
            "</table></div>")

# ---------- feature block -----------------------------------------------
def feature_block(fid:int)->str:
    F=h5_chat["scores"].shape[0]
    if fid<0 or fid>=F: return f"<p style='color:red'>Feature {fid} out of range.</p>"

    lbl = LABELS.get(_lbl_key(current_layer, current_trainer, fid), {})
    interesting = lbl.get("interesting", False)
    notes_txt   = lbl.get("notes", "")

    checked = "checked" if interesting else ""
    head_cls = "interesting" if interesting else ""

    head=(f"<div class='feat-header {head_cls}' id='feat-{fid}'>"
          f"<div><h2>Feature {fid}</h2><span class='meta'>(layer {current_layer}, trainer {current_trainer})</span></div>"
          f"<div class='feat-controls'>"
          f"<label><input type='checkbox' class='interesting' id='chk-{fid}' {checked}> interesting</label>&nbsp;"
          f"<textarea class='note' id='note-{fid}' rows='2' placeholder='notes'>{html.escape(notes_txt, False)}</textarea>"
          "</div></div>")
    def section(name,h5file,css_id):
        tok,acts,score=h5file["tokens"][fid][:TOP_N],h5file["sae_acts"][fid][:TOP_N],h5file["scores"][fid][:TOP_N]
        freq,total=h5file["frequency"][fid],h5file.attrs["tokens_seen"]; freq=sci(freq/total if total else 0.0)
        hdr=f"<div class='sec-header' onclick=\"document.getElementById('{css_id}').classList.toggle('collapsed')\"><b>{name}</b>&nbsp;<span class='freq'>({freq})</span></div>"
        body=f"<div id='{css_id}' class='sec-body collapsed'>"+"".join(example_html(tok[k],acts[k],score[k]) for k in range(len(tok)))+"</div>"
        return hdr+body
    html_rows=[head,sim_panel(fid),"<hr/>",section("CHAT",h5_chat,f"chat-sec-{fid}"),section("PRETRAIN",h5_pt,f"pt-sec-{fid}")]
    return "<div class='feature'>"+"\n".join(html_rows)+"</div>"

# ---------- Flask routes -------------------------------------------------
@app.route("/")
def index():
    return Response(f"""
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>SAE Feature Explorer</title>
<style>
 /* (same CSS as before – unchanged) */
 @import url("https://fonts.googleapis.com/css2?family=Noto+Sans+Mono:wght@400&display=swap");
 body{{font-family:'Noto Sans Mono',monospace;margin:20px;}}
 #ctrl{{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;margin-bottom:16px;gap:12px;}}
 #ctrl .left, #ctrl .right{{display:flex;align-items:flex-end;gap:8px;flex-wrap:wrap;}}
 input[type=number]{{width:120px;font-size:1em;}}
input.note, textarea.note {{
  width: 320px;
  font-family: inherit;
  font-size: 0.9em;
}}
textarea.note {{
  resize: vertical;
  line-height: 1.3;
}}
 .feature{{margin-bottom:32px;}}
 .feat-header{{display:flex;justify-content:space-between;align-items:flex-end;border-left:4px solid transparent;padding-left:4px;}}
 .feat-header.interesting{{background:#fff7d1;border-left-color:#d84400;}}
 .feat-header.interesting h2{{font-weight:700;}}
 .feat-controls{{font-size:0.9em;}}
 .sim{{table-layout:fixed;width:100%;border-collapse:collapse;font-size:0.9em;}}
 .sim th,.sim td{{width:50%;border:1px solid #ccc;padding:2px 4px;vertical-align:top;}}
 .token-strip{{display:block;white-space:nowrap;overflow-x:auto;padding:2px;border-radius:4px;}}
 .token-strip.top{{background:#e8f9e8;}} .token-strip.bottom{{background:#fbeaea;}}
 .pill{{display:inline-block;padding:1px 4px;margin:1px;border:1px solid #bbb;border-radius:6px;}}
 .sec-header{{cursor:pointer;background:#eee;padding:6px 8px;border:1px solid #ccc;border-radius:4px;margin-top:12px;}}
 .sec-header:hover{{background:#ddd;}} .freq{{color:#555;font-size:0.85em;}}
 .sec-body{{border:1px solid #ccc;border-top:none;padding:8px;}}
 .sec-body.collapsed{{display:none;}}
 .example{{border:1px solid #c3c3c3;margin:14px 0;padding:10px;border-radius:6px;background:#fafafa;}}
 .snippet{{display:flex;white-space:pre;overflow-x:auto;}}
 .mono-window{{white-space:pre;}}
 .peak{{background:#d84400;color:#fff;border-radius:8px;padding:0;}}
 .preview-tok{{padding:0;}} .score{{margin-left:auto;color:#555;font-size:0.9em;}}
 .tok{{display:inline-block;padding:0 1px;border-right:1px solid rgba(0,0,0,0.08);}}
 .tok:last-child{{border-right:none;}} .tok:hover{{border-right-color:#888;}}
 .full{{display:none;white-space:pre-wrap;margin-top:12px;padding-top:6px;border-top:1px dashed #ccc;}}
 .example.collapsed .full{{display:none;}} .example:not(.collapsed) .full{{display:block;}}
</style>
</head><body>
<div id="ctrl">
  <div class="left">
    <label>Features:&nbsp;<input id="fids" type="text" style="width:260px"
             placeholder="e.g. 3, 17  42" onkeyup="if(event.key==='Enter') load();"></label>
    <button onclick="load()">Load</button>
    <button onclick="saveLabels()">Save Labels</button>
  </div>
  <div class="right">
    <label>Layer:<input id="layer" type="number" min="0" step="1" value="{current_layer}"></label>
    <label>Trainer:<input id="trainer" type="number" min="0" step="1" value="{current_trainer}"></label>
    <button onclick="setVariant()">Apply</button>
  </div>
</div>
<div id="out"></div>

<script>
// --- helper: keep values/checked attrs & textarea text so saved HTML preserves them
function persistAttr(el){{
  if(el.type==='checkbox'){{
      el.checked ? el.setAttribute('checked','') : el.removeAttribute('checked');
  }}else if(el.tagName==='TEXTAREA'){{
      el.setAttribute('value',el.value);
      el.textContent = el.value;          // ensure note text is in static HTML
  }}else{{
      el.setAttribute('value',el.value);
  }}
}}

// --- load features ---------------------------------------------------
async function load(){{
  const raw=document.getElementById('fids').value.trim();
  if(!raw) return;
  const ids=raw.split(/[,\\s]+/).filter(Boolean);
  const out=document.getElementById('out'); out.innerHTML='<p>Loading…</p>';
  const blocks=await Promise.all(ids.map(id=>fetch(`/feature/${{id}}`).then(r=>r.text())));
  out.innerHTML=blocks.join('');
  // wire up handlers & mirror attributes
  ids.forEach(id=>{{
     const chk=document.getElementById('chk-'+id);
     const note=document.getElementById('note-'+id);
     const feat=document.getElementById('feat-'+id);
     chk.onchange=()=>{{ persistAttr(chk); feat.classList.toggle('interesting',chk.checked); }};
     note.oninput=()=>persistAttr(note);
     persistAttr(chk); persistAttr(note);
  }});
}}

// --- variant switch --------------------------------------------------
async function setVariant(){{
  const layer=document.getElementById('layer').value;
  const trainer=document.getElementById('trainer').value;
  if(layer===''||trainer==='') return;
  const out=document.getElementById('out'); out.innerHTML='<p>Loading…</p>';
  const msg=await (await fetch(`/variant/${{layer}}/${{trainer}}`)).text();
  out.innerHTML=msg;
}}

// --- save labels ----------------------------------------------------
async function saveLabels(){{
  const layer = Number(document.getElementById('layer').value);
  const trainer = Number(document.getElementById('trainer').value);
  const feats=document.querySelectorAll('.feat-header');
  const data=[];
  feats.forEach(fh=>{{
    const fid=parseInt(fh.id.replace('feat-',''));
    const chk=document.getElementById('chk-'+fid);
    const note=document.getElementById('note-'+fid);
    const interesting=chk.checked;
    const notes=note.value;
    if(interesting || notes.trim()!==""){{   // only send if something to save
        data.push({{layer, trainer, fid, interesting, notes}});
    }}
  }});
  const res=await fetch('/labels',{{method:'POST',
       headers:{{'Content-Type':'application/json'}},
       body:JSON.stringify(data)}});
  alert(await res.text());
}}
</script>
</body></html>
""", mimetype="text/html")

@app.route("/feature/<int:fid>")
def feature(fid:int):
    return Response(feature_block(fid), mimetype="text/html")

@app.route("/variant/<int:layer>/<int:trainer>")
def variant(layer:int, trainer:int):
    try:
        switch_variant(layer, trainer)
        msg=f"<p style='color:green'>Switched to layer {layer}, trainer {trainer}</p>"
    except FileNotFoundError:
        msg=f"<p style='color:red'>Files not found for layer {layer}, trainer {trainer}</p>"
    return Response(msg, mimetype="text/html")

# ---------- label save API ---------------------------------------------
@app.route("/labels", methods=["POST"])
def save_labels_api():
    updates=request.get_json(force=True, silent=True) or []
    for up in updates:
        key=_lbl_key(int(up["layer"]), int(up["trainer"]), int(up["fid"]))
        if up.get("interesting") or up.get("notes","").strip():
            LABELS[key]={"interesting":bool(up["interesting"]),
                          "notes":up.get("notes","")}
        elif key in LABELS:            # neither interesting nor notes → drop
            LABELS.pop(key)
    _save_labels()
    return f"Saved {len(updates)} label(s).", 200

# ---------- run ---------------------------------------------------------
if __name__=="__main__":
    print(f" * running on http://localhost:{PORT}/")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

#!/usr/bin/env python3
import io, json, os, re, uuid
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request
from werkzeug.utils import secure_filename
import anthropic
 
# ── Optional parsers ──────────────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
 
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
 
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
 
# ── Config ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
 
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_PASSWORD    = os.environ.get("ADMIN_PASSWORD", "debate2024")
BOT_NAME          = os.environ.get("BOT_NAME", "Debate Assistant")
KB_FILE           = os.environ.get("KB_FILE", "knowledge_base.json")
MODEL             = "claude-haiku-4-5-20251001"
 
# ── Knowledge base ────────────────────────────────────────────────────────────
def load_kb():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "chunks": []}
 
def save_kb(kb):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
 
knowledge_base = load_kb()
 
# ── Text extraction ───────────────────────────────────────────────────────────
def extract_pdf(file_bytes):
    if HAS_PDFPLUMBER:
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                parts = []
                text = page.extract_text()
                if text and text.strip():
                    parts.append(text.strip())
                # Extract tables as readable text
                for table in (page.extract_tables() or []):
                    if not table:
                        continue
                    headers = [str(c).strip() if c else "" for c in table[0]]
                    has_headers = any(h and not h.replace(" ","").isnumeric() for h in headers)
                    rows = []
                    for ri, row in enumerate(table):
                        if ri == 0 and has_headers:
                            continue
                        cells = [str(c).strip() if c else "" for c in row]
                        if not any(cells):
                            continue
                        if has_headers:
                            rows.append(" | ".join(
                                f"{h}: {v}" for h, v in zip(headers, cells) if h or v
                            ))
                        else:
                            rows.append(" | ".join(cells))
                    if rows:
                        parts.append("TABLE:\n" + "\n".join(rows))
                if parts:
                    pages.append(f"[Page {i+1}]\n" + "\n\n".join(parts))
        return "\n\n".join(pages)
    elif HAS_FITZ:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {i+1}]\n{text}")
        return "\n\n".join(pages)
    else:
        raise ValueError("No PDF library available. Run: pip install pdfplumber")
 
def extract_docx(file_bytes):
    if not HAS_DOCX:
        raise ValueError("python-docx not installed")
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
 
def extract_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="replace")
 
# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=500, overlap=60):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = []
    for para in paragraphs:
        words = para.split()
        if len(current) + len(words) <= chunk_size:
            current.extend(words)
        else:
            if current:
                chunks.append(" ".join(current))
            if len(words) > chunk_size:
                for i in range(0, len(words), chunk_size - overlap):
                    seg = " ".join(words[i:i + chunk_size])
                    if seg:
                        chunks.append(seg)
                current = words[-overlap:]
            else:
                current = words
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 10]
 
# ── Retrieval ─────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","on","at","by","for","with","from","this",
    "that","it","its","what","how","when","where","who","which","and","or",
    "but","if","then","than","as","so","yet","both","either","neither","not",
    "no","nor","just","because","although","though","while","i","you","he",
    "she","we","they","me","him","her","us","them","about","also","any",
    "each","into","through","during","before","after","above","below","up",
    "down","out","off","over","under","again","further","there","here",
}
 
DEBATE_SYNONYMS = {
    "counterplan":["counter plan","counter-plan","cp"],
    "counter plan":["counterplan","counter-plan","cp"],
    "cp":["counterplan","counter plan"],
    "disadvantage":["disad","da","disadvantages"],
    "disad":["disadvantage","da"],
    "da":["disadvantage","disad"],
    "kritik":["critique","k","critical argument"],
    "critique":["kritik","k"],
    "k":["kritik","critique"],
    "topicality":["t","on-topic","topical"],
    "t":["topicality"],
    "permutation":["perm","permutations"],
    "perm":["permutation"],
    "solvency":["solve","solves"],
    "framework":["fw","value framework","criterion"],
    "fw":["framework"],
    "evidence":["ev","card","cards"],
    "ev":["evidence","card"],
    "card":["evidence","ev"],
    "affirmative":["aff","pro"],
    "aff":["affirmative","pro"],
    "negative":["neg","con"],
    "neg":["negative","con"],
    "public forum":["pf","pfd"],
    "pf":["public forum","public forum debate"],
    "lincoln douglas":["ld","lincoln-douglas"],
    "ld":["lincoln douglas","lincoln-douglas"],
    "policy debate":["cx","policy"],
    "cx":["policy debate","policy"],
    "congressional":["congress","student congress"],
    "congress":["congressional","student congress"],
    "extemp":["extemporaneous","extemporaneous speaking"],
    "extemporaneous":["extemp"],
    "crossfire":["cross examination","cross-ex"],
    "cross examination":["crossfire","cross-ex","cx"],
    "flow":["flowing","flowed"],
    "rebuttal":["rebuttals","rebut"],
    "contention":["contentions","argument"],
    "speaker points":["speaks","speaker point"],
    "speaks":["speaker points","speaker point"],
    "ballot":["ballots","decision"],
    "resolution":["res","topic"],
    "res":["resolution","topic"],
    "prep time":["prep","preparation time"],
    "prep":["prep time","preparation time"],
}
 
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"(\w)-(\w)", r"\1\2", text)
    return text
 
def stem_word(word):
    if len(word) <= 4:
        return word
    for suffix in ["ations","ation","ings","ing","tions","tion",
                   "ness","ments","ment","ers","ies","ed","es","s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word
 
def get_keywords(text):
    words = re.findall(r"\b[a-z]{2,}\b", normalize_text(text))
    return {w for w in words if w not in STOP_WORDS}
 
def expand_query(query):
    expanded = set()
    q_lower = query.lower()
    q_norm  = normalize_text(query)
    base = get_keywords(q_lower) | get_keywords(q_norm)
    expanded.update(base)
    expanded.update(stem_word(w) for w in base)
    for word in list(base):
        if word in DEBATE_SYNONYMS:
            for syn in DEBATE_SYNONYMS[word]:
                expanded.update(get_keywords(syn))
                expanded.add(normalize_text(syn).replace(" ", ""))
    for phrase, synonyms in DEBATE_SYNONYMS.items():
        if phrase in q_lower or phrase in q_norm:
            for syn in synonyms:
                expanded.update(get_keywords(syn))
    expanded.add(re.sub(r"\s+", "", q_norm))
    return expanded - STOP_WORDS
 
def search(query, chunks, top_k=6):
    if not chunks:
        return []
    q_kw = expand_query(query)
    if not q_kw:
        return chunks[:top_k]
    scored = []
    for chunk in chunks:
        raw   = chunk["text"]
        norm  = normalize_text(raw)
        words = re.findall(r"\b[a-z]{2,}\b", norm)
        dlen  = len(words) + 1
        c_kw  = {w for w in words if w not in STOP_WORDS}
        c_all = c_kw | {stem_word(w) for w in c_kw}
        score = 0.0
        for kw in q_kw:
            ks = stem_word(kw)
            if kw in c_all or ks in c_all:
                tf = (words.count(kw) + words.count(ks)) / dlen
                score += (tf + 0.01) * (1 + 1 / max(len(kw) ** 0.3, 1))
        for kw in q_kw:
            if len(kw) > 5 and kw in norm:
                score += 0.2
        q_lower = query.lower()
        for phrase in DEBATE_SYNONYMS:
            if len(phrase) > 4 and phrase in q_lower and phrase in norm:
                score += 0.4
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
 
# ── HTML template ─────────────────────────────────────────────────────────────
HTML = (
    "<!DOCTYPE html>"
    "<html lang='en'>"
    "<head>"
    "<meta charset='UTF-8'/>"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'/>"
    "<title>Debate AI Assistant</title>"
    "<style>"
    "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}"
    ":root{--navy:#1a2744;--navy2:#243460;--gold:#c9a227;--gold2:#e0b93a;"
    "--light:#f4f6fb;--white:#ffffff;--gray:#6b7280;--border:#dde3f0;"
    "--red:#dc2626;--green:#16a34a;--radius:12px;--shadow:0 4px 24px rgba(26,39,68,.12)}"
    "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
    "background:var(--light);color:#1f2937;min-height:100vh;display:flex;flex-direction:column}"
    "header{background:var(--navy);color:var(--white);padding:0 24px;height:64px;"
    "display:flex;align-items:center;justify-content:space-between;"
    "box-shadow:0 2px 12px rgba(0,0,0,.25);position:sticky;top:0;z-index:100}"
    ".header-left{display:flex;align-items:center;gap:12px}"
    ".logo-icon{width:38px;height:38px;background:var(--gold);border-radius:8px;"
    "display:flex;align-items:center;justify-content:center;font-size:20px}"
    ".brand h1{font-size:18px;font-weight:700;letter-spacing:-.3px}"
    ".brand p{font-size:12px;opacity:.7;margin-top:1px}"
    ".header-right{display:flex;align-items:center;gap:10px}"
    ".status-pill{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,.1);"
    "border-radius:20px;padding:4px 12px;font-size:12px}"
    ".status-dot{width:7px;height:7px;border-radius:50%;background:#9ca3af}"
    ".status-dot.online{background:#34d399}"
    ".status-dot.offline{background:var(--red)}"
    ".admin-btn{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);"
    "color:var(--white);border-radius:8px;padding:6px 14px;font-size:13px;cursor:pointer;"
    "display:flex;align-items:center;gap:6px;transition:background .2s}"
    ".admin-btn:hover{background:rgba(255,255,255,.25)}"
    ".main{display:flex;flex:1;gap:0;overflow:hidden}"
    ".admin-panel{width:340px;min-width:340px;background:var(--white);"
    "border-right:1px solid var(--border);display:none;flex-direction:column;overflow:hidden}"
    ".admin-panel.open{display:flex}"
    ".admin-header{background:var(--navy2);color:var(--white);padding:16px 20px;"
    "font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px}"
    ".admin-body{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:20px}"
    ".admin-section-title{font-size:11px;font-weight:700;text-transform:uppercase;"
    "letter-spacing:.8px;color:var(--gray);margin-bottom:8px}"
    ".lock-screen{display:flex;flex-direction:column;gap:10px}"
    ".lock-screen p{font-size:13px;color:var(--gray)}"
    "label{font-size:13px;font-weight:500;display:block;margin-bottom:4px}"
    "input[type=password],input[type=text]{width:100%;padding:9px 12px;"
    "border:1px solid var(--border);border-radius:8px;font-size:14px;outline:none;transition:border .2s}"
    "input:focus{border-color:var(--navy2)}"
    ".btn{padding:9px 18px;border-radius:8px;border:none;font-size:14px;font-weight:600;"
    "cursor:pointer;transition:background .2s,transform .1s}"
    ".btn:active{transform:scale(.97)}"
    ".btn-primary{background:var(--navy);color:var(--white)}"
    ".btn-primary:hover{background:var(--navy2)}"
    ".btn-danger{background:#fee2e2;color:var(--red)}"
    ".btn-danger:hover{background:#fecaca}"
    ".btn-sm{padding:5px 12px;font-size:12px}"
    ".btn-block{width:100%}"
    ".upload-zone{border:2px dashed var(--border);border-radius:var(--radius);"
    "padding:24px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;position:relative}"
    ".upload-zone:hover,.upload-zone.drag{border-color:var(--navy2);background:#f0f4ff}"
    ".upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}"
    ".upload-zone .upload-icon{font-size:32px;margin-bottom:8px}"
    ".upload-zone p{font-size:13px;color:var(--gray)}"
    ".upload-zone strong{color:var(--navy)}"
    ".progress-bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden;display:none;margin-top:8px}"
    ".progress-bar-fill{height:100%;background:var(--gold);border-radius:3px;width:0%;transition:width .3s}"
    ".doc-list{display:flex;flex-direction:column;gap:8px}"
    ".doc-item{background:var(--light);border:1px solid var(--border);border-radius:8px;"
    "padding:10px 12px;display:flex;align-items:flex-start;justify-content:space-between;gap:8px}"
    ".doc-info{flex:1;min-width:0}"
    ".doc-name{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
    ".doc-meta{font-size:11px;color:var(--gray);margin-top:2px}"
    ".doc-icon{font-size:18px;margin-right:6px;flex-shrink:0}"
    ".doc-left{display:flex;align-items:center;flex:1;min-width:0}"
    ".no-docs{font-size:13px;color:var(--gray);text-align:center;padding:16px 0}"
    ".alert{padding:10px 14px;border-radius:8px;font-size:13px;display:none}"
    ".alert.show{display:block}"
    ".alert-success{background:#dcfce7;color:var(--green);border:1px solid #bbf7d0}"
    ".alert-error{background:#fee2e2;color:var(--red);border:1px solid #fecaca}"
    ".chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}"
    ".chat-messages{flex:1;overflow-y:auto;padding:24px 20px;"
    "display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth}"
    ".welcome-card{max-width:520px;margin:20px auto;background:var(--white);"
    "border-radius:var(--radius);padding:32px;text-align:center;"
    "box-shadow:var(--shadow);border:1px solid var(--border)}"
    ".welcome-card .big-icon{font-size:48px;margin-bottom:12px}"
    ".welcome-card h2{font-size:22px;color:var(--navy);margin-bottom:8px}"
    ".welcome-card p{font-size:14px;color:var(--gray);line-height:1.6}"
    ".welcome-card .tips{margin-top:20px;text-align:left;background:var(--light);"
    "border-radius:8px;padding:14px 16px}"
    ".welcome-card .tips p{font-size:13px;margin-bottom:4px}"
    ".tip-label{font-weight:700;color:var(--navy)}"
    ".message{display:flex;gap:10px;align-items:flex-start;max-width:800px}"
    ".message.user{flex-direction:row-reverse;margin-left:auto}"
    ".message.bot{margin-right:auto}"
    ".avatar{width:34px;height:34px;border-radius:50%;flex-shrink:0;"
    "display:flex;align-items:center;justify-content:center;font-size:16px}"
    ".message.user .avatar{background:var(--navy);color:var(--white);font-size:13px;font-weight:700}"
    ".message.bot .avatar{background:var(--gold);color:var(--navy)}"
    ".bubble{padding:12px 16px;border-radius:var(--radius);font-size:14px;"
    "line-height:1.6;max-width:100%;word-break:break-word}"
    ".message.user .bubble{background:var(--navy);color:var(--white);border-top-right-radius:4px}"
    ".message.bot .bubble{background:var(--white);border:1px solid var(--border);"
    "border-top-left-radius:4px;box-shadow:0 1px 6px rgba(0,0,0,.06)}"
    ".sources-tag{display:flex;align-items:center;gap:6px;margin-top:8px;flex-wrap:wrap}"
    ".source-label{font-size:11px;color:var(--gray);font-weight:600}"
    ".source-chip{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;"
    "border-radius:20px;padding:2px 10px;font-size:11px;white-space:nowrap;"
    "overflow:hidden;text-overflow:ellipsis;max-width:180px}"
    ".typing-bubble{background:var(--white);border:1px solid var(--border);"
    "border-radius:var(--radius);border-top-left-radius:4px;padding:14px 18px;"
    "display:flex;gap:5px;align-items:center;box-shadow:0 1px 6px rgba(0,0,0,.06)}"
    ".dot{width:7px;height:7px;border-radius:50%;background:#9ca3af;animation:bounce 1.2s infinite}"
    ".dot:nth-child(2){animation-delay:.2s}"
    ".dot:nth-child(3){animation-delay:.4s}"
    "@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-8px)}}"
    ".input-bar{background:var(--white);border-top:1px solid var(--border);padding:16px 20px}"
    ".input-wrap{display:flex;gap:10px;align-items:flex-end;background:var(--light);"
    "border:1.5px solid var(--border);border-radius:12px;padding:10px 14px;transition:border-color .2s}"
    ".input-wrap:focus-within{border-color:var(--navy2);background:var(--white)}"
    "#user-input{flex:1;border:none;background:transparent;font-size:14px;line-height:1.5;"
    "resize:none;outline:none;max-height:120px;min-height:24px;font-family:inherit}"
    ".send-btn{background:var(--navy);color:var(--white);border:none;border-radius:8px;"
    "width:36px;height:36px;cursor:pointer;display:flex;align-items:center;justify-content:center;"
    "flex-shrink:0;transition:background .2s;font-size:16px}"
    ".send-btn:hover:not(:disabled){background:var(--navy2)}"
    ".send-btn:disabled{opacity:.4;cursor:not-allowed}"
    ".input-hint{font-size:11px;color:var(--gray);margin-top:6px;text-align:center}"
    ".no-kb-banner{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;"
    "padding:10px 16px;font-size:13px;color:#92400e;margin:0 20px;"
    "display:flex;align-items:center;gap:8px}"
    ".no-kb-banner.hidden{display:none}"
    "@media(max-width:640px){.admin-panel{width:100%;min-width:unset;position:absolute;"
    "z-index:50;height:100%;top:64px;left:0}.brand p{display:none}}"
    "</style>"
    "</head>"
    "<body>"
    "<header>"
    "<div class='header-left'>"
    "<div class='logo-icon'>&#9878;</div>"
    "<div class='brand'>"
    "<h1 id='bot-name-header'>Debate Assistant</h1>"
    "<p>AI Knowledge Base &middot; Coaches &amp; Judges</p>"
    "</div></div>"
    "<div class='header-right'>"
    "<div class='status-pill'>"
    "<div class='status-dot' id='status-dot'></div>"
    "<span id='status-text'>Connecting&hellip;</span>"
    "</div>"
    "<button class='admin-btn' onclick='toggleAdmin()'>&#9881; Admin</button>"
    "</div></header>"
    "<div class='main'>"
    "<aside class='admin-panel' id='admin-panel'>"
    "<div class='admin-header'>&#128274; Admin Panel</div>"
    "<div class='admin-body'>"
    "<div id='lock-screen'>"
    "<div class='admin-section-title'>Authentication</div>"
    "<div class='lock-screen'>"
    "<p>Enter your admin password to manage the knowledge base.</p>"
    "<div><label for='admin-pw'>Password</label>"
    "<input type='password' id='admin-pw' placeholder='Enter password&hellip;'"
    " onkeydown='if(event.key===\"Enter\")verifyAdmin()'/></div>"
    "<button class='btn btn-primary btn-block' onclick='verifyAdmin()'>Unlock &rarr;</button>"
    "<div class='alert alert-error' id='pw-error'>Incorrect password. Try again.</div>"
    "</div></div>"
    "<div id='admin-content' style='display:none;flex-direction:column;gap:20px'>"
    "<div><div class='admin-section-title'>Upload Documents</div>"
    "<div class='upload-zone' id='upload-zone'"
    " ondragover=\"event.preventDefault();this.classList.add('drag')\""
    " ondragleave=\"this.classList.remove('drag')\" ondrop='handleDrop(event)'>"
    "<input type='file' id='file-input' multiple accept='.pdf,.docx,.txt'"
    " onchange='handleFileSelect(event)'/>"
    "<div class='upload-icon'>&#128193;</div>"
    "<p><strong>Click or drag files here</strong></p>"
    "<p>PDF, DOCX, TXT up to 50 MB</p>"
    "</div>"
    "<div class='progress-bar' id='progress-bar'>"
    "<div class='progress-bar-fill' id='progress-fill'></div></div>"
    "<div class='alert alert-success' id='upload-success'></div>"
    "<div class='alert alert-error' id='upload-error'></div>"
    "</div>"
    "<div><div class='admin-section-title'>Knowledge Base (<span id='doc-count'>0</span> docs)</div>"
    "<div class='doc-list' id='doc-list'><p class='no-docs'>No documents uploaded yet.</p></div>"
    "</div>"
    "<button class='btn btn-danger btn-sm btn-block' onclick='signOutAdmin()'>Sign Out</button>"
    "</div></div></aside>"
    "<section class='chat-area'>"
    "<div class='no-kb-banner hidden' id='no-kb-banner'>"
    "&#9888; No documents in the knowledge base yet. Open Admin to upload files."
    "</div>"
    "<div class='chat-messages' id='chat-messages'>"
    "<div class='welcome-card' id='welcome-card'>"
    "<div class='big-icon'>&#9878;</div>"
    "<h2 id='bot-name-welcome'>Debate Assistant</h2>"
    "<p>Your AI-powered resource for high school debate rules, procedures, and best practices."
    " Ask me anything &mdash; I will answer only from your uploaded knowledge base.</p>"
    "<div class='tips'>"
    "<p><span class='tip-label'>Try asking:</span></p>"
    "<p>&bull; What are the speaker point guidelines?</p>"
    "<p>&bull; How does time allocation work in LD?</p>"
    "<p>&bull; What is a topicality argument?</p>"
    "<p>&bull; Can the Aff run a CP?</p>"
    "</div></div></div>"
    "<div class='input-bar'>"
    "<div class='input-wrap'>"
    "<textarea id='user-input' rows='1' placeholder='Ask a debate question&hellip;'"
    " onkeydown='handleKey(event)' oninput='autoResize(this)'></textarea>"
    "<button class='send-btn' id='send-btn' onclick='sendMessage()'>&#9658;</button>"
    "</div>"
    "<div class='input-hint'>Answers come only from your uploaded documents &middot; Press Enter to send</div>"
    "</div></section></div>"
    "<script>"
    "var messages=[];"
    "var adminUnlocked=false;"
    "var adminPW='';"
    "async function init(){"
    "  try{"
    "    var r=await fetch('/api/status');"
    "    var s=await r.json();"
    "    document.getElementById('bot-name-header').textContent=s.bot_name;"
    "    document.getElementById('bot-name-welcome').textContent=s.bot_name;"
    "    var dot=document.getElementById('status-dot');"
    "    var txt=document.getElementById('status-text');"
    "    if(s.has_api_key){dot.className='status-dot online';txt.textContent='Online';}"
    "    else{dot.className='status-dot offline';txt.textContent='No API key';}"
    "    if(s.document_count===0)document.getElementById('no-kb-banner').classList.remove('hidden');"
    "    document.getElementById('doc-count').textContent=s.document_count;"
    "  }catch(e){console.error(e);}"
    "}"
    "function toggleAdmin(){"
    "  var p=document.getElementById('admin-panel');"
    "  p.classList.toggle('open');"
    "  if(p.classList.contains('open')&&adminUnlocked)loadDocuments();"
    "}"
    "async function verifyAdmin(){"
    "  var pw=document.getElementById('admin-pw').value;"
    "  try{"
    "    var r=await fetch('/api/verify-admin',{method:'POST',"
    "      headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});"
    "    var d=await r.json();"
    "    if(d.valid){"
    "      adminPW=pw;adminUnlocked=true;"
    "      document.getElementById('lock-screen').style.display='none';"
    "      document.getElementById('admin-content').style.display='flex';"
    "      loadDocuments();"
    "    }else{showAlert('pw-error',true);}"
    "  }catch(e){showAlert('pw-error',true);}"
    "}"
    "function signOutAdmin(){"
    "  adminUnlocked=false;adminPW='';"
    "  document.getElementById('lock-screen').style.display='block';"
    "  document.getElementById('admin-content').style.display='none';"
    "  document.getElementById('admin-pw').value='';"
    "}"
    "async function loadDocuments(){"
    "  try{"
    "    var r=await fetch('/api/documents');"
    "    var docs=await r.json();"
    "    renderDocList(docs);"
    "    document.getElementById('doc-count').textContent=docs.length;"
    "    var b=document.getElementById('no-kb-banner');"
    "    if(docs.length===0)b.classList.remove('hidden');"
    "    else b.classList.add('hidden');"
    "  }catch(e){console.error(e);}"
    "}"
    "function renderDocList(docs){"
    "  var el=document.getElementById('doc-list');"
    "  if(!docs.length){el.innerHTML='<p class=\"no-docs\">No documents uploaded yet.</p>';return;}"
    "  el.innerHTML=docs.map(function(d){"
    "    var ext=d.name.split('.').pop().toLowerCase();"
    "    var icon=ext==='pdf'?'&#128196;':ext==='docx'?'&#128221;':'&#128203;';"
    "    var date=new Date(d.uploaded_at).toLocaleDateString();"
    "    return '<div class=\"doc-item\"><div class=\"doc-left\"><span class=\"doc-icon\">'+icon+'</span>'"
    "    +'<div class=\"doc-info\"><div class=\"doc-name\" title=\"'+d.name+'\">'+d.name+'</div>'"
    "    +'<div class=\"doc-meta\">'+d.chunk_count+' chunks &middot; '+date+'</div></div></div>'"
    "    +'<button class=\"btn btn-danger btn-sm\" onclick=\"deleteDoc(\\'' +d.id+ '\\',\\'' +d.name+ \\')\">&#10005;</button></div>';"
    "  }).join('');"
    "}"
    "async function deleteDoc(id,name){"
    "  if(!confirm('Remove '+name+' from the knowledge base?'))return;"
    "  try{"
    "    var r=await fetch('/api/documents/'+id,{method:'DELETE',"
    "      headers:{'Content-Type':'application/json'},body:JSON.stringify({password:adminPW})});"
    "    var d=await r.json();"
    "    if(d.success)loadDocuments();"
    "  }catch(e){alert('Delete failed.');}"
    "}"
    "function handleFileSelect(e){uploadFiles(e.target.files);}"
    "function handleDrop(e){"
    "  e.preventDefault();"
    "  document.getElementById('upload-zone').classList.remove('drag');"
    "  uploadFiles(e.dataTransfer.files);"
    "}"
    "async function uploadFiles(fileList){"
    "  for(var i=0;i<fileList.length;i++){await uploadFile(fileList[i]);}"
    "  loadDocuments();"
    "  document.getElementById('file-input').value='';"
    "}"
    "async function uploadFile(file){"
    "  var allowed=['pdf','docx','txt'];"
    "  var ext=file.name.split('.').pop().toLowerCase();"
    "  if(allowed.indexOf(ext)<0){showAlert('upload-error','Unsupported file type. Use PDF, DOCX, or TXT.');return;}"
    "  var bar=document.getElementById('progress-bar');"
    "  var fill=document.getElementById('progress-fill');"
    "  bar.style.display='block';fill.style.width='20%';"
    "  var form=new FormData();"
    "  form.append('file',file,file.name);"
    "  form.append('password',adminPW);"
    "  try{"
    "    fill.style.width='60%';"
    "    var r=await fetch('/api/upload',{method:'POST',body:form});"
    "    fill.style.width='100%';"
    "    var d=await r.json();"
    "    setTimeout(function(){bar.style.display='none';fill.style.width='0%';},800);"
    "    if(d.success)showAlert('upload-success','Uploaded '+file.name+' ('+d.chunks_created+' chunks)');"
    "    else showAlert('upload-error',d.error||'Upload failed.');"
    "  }catch(e){bar.style.display='none';showAlert('upload-error','Upload failed.');}"
    "}"
    "function handleKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}}"
    "function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px';}"
    "async function sendMessage(){"
    "  var input=document.getElementById('user-input');"
    "  var text=input.value.trim();"
    "  if(!text)return;"
    "  var w=document.getElementById('welcome-card');"
    "  if(w)w.remove();"
    "  input.value='';input.style.height='auto';"
    "  messages.push({role:'user',content:text});"
    "  appendMessage('user',text,[]);"
    "  var btn=document.getElementById('send-btn');"
    "  btn.disabled=true;"
    "  var tid='typing-'+Date.now();"
    "  appendTyping(tid);"
    "  try{"
    "    var r=await fetch('/api/chat',{method:'POST',"
    "      headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:messages})});"
    "    var d=await r.json();"
    "    removeTyping(tid);"
    "    if(d.error)appendMessage('bot','Error: '+d.error,[]);"
    "    else{messages.push({role:'assistant',content:d.response});appendMessage('bot',d.response,d.sources||[]);}"
    "  }catch(e){removeTyping(tid);appendMessage('bot','Could not reach the server.',[]);}"
    "  btn.disabled=false;input.focus();"
    "}"
    "function appendMessage(role,text,sources){"
    "  var c=document.getElementById('chat-messages');"
    "  var div=document.createElement('div');"
    "  div.className='message '+role;"
    "  var av=document.createElement('div');av.className='avatar';"
    "  av.textContent=role==='user'?'You':'\\u2696';"
    "  var bub=document.createElement('div');bub.className='bubble';"
    "  bub.innerHTML=esc(text).replace(/\\n/g,'<br>');"
    "  if(sources&&sources.length){"
    "    var u=[...new Set(sources)];"
    "    var sd=document.createElement('div');sd.className='sources-tag';"
    "    sd.innerHTML='<span class=\"source-label\">Sources:</span>'"
    "    +u.map(function(s){return '<span class=\"source-chip\" title=\"'+s+'\">'+s+'</span>';}).join('');"
    "    bub.appendChild(sd);"
    "  }"
    "  div.appendChild(av);div.appendChild(bub);"
    "  c.appendChild(div);c.scrollTop=c.scrollHeight;"
    "}"
    "function appendTyping(id){"
    "  var c=document.getElementById('chat-messages');"
    "  var div=document.createElement('div');"
    "  div.className='message bot';div.id=id;"
    "  div.innerHTML='<div class=\"avatar\">\\u2696</div>'"
    "  +'<div class=\"typing-bubble\"><div class=\"dot\"></div><div class=\"dot\"></div><div class=\"dot\"></div></div>';"
    "  c.appendChild(div);c.scrollTop=c.scrollHeight;"
    "}"
    "function removeTyping(id){var el=document.getElementById(id);if(el)el.remove();}"
    "function esc(s){"
    "  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');"
    "}"
    "function showAlert(id,msg){"
    "  var el=document.getElementById(id);if(!el)return;"
    "  if(typeof msg==='string')el.textContent=msg;"
    "  el.classList.add('show');"
    "  setTimeout(function(){el.classList.remove('show');},5000);"
    "}"
    "init();"
    "</script>"
    "</body></html>"
)
 
# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)
 
@app.route("/api/status")
def status():
    return jsonify({
        "bot_name":       BOT_NAME,
        "has_api_key":    bool(ANTHROPIC_API_KEY),
        "document_count": len(knowledge_base.get("documents", [])),
        "chunk_count":    len(knowledge_base.get("chunks", [])),
    })
 
@app.route("/api/verify-admin", methods=["POST"])
def verify_admin():
    pw = (request.json or {}).get("password", "")
    return jsonify({"valid": pw == ADMIN_PASSWORD})
 
@app.route("/api/upload", methods=["POST"])
def upload():
    global knowledge_base
    pw = request.form.get("password", "")
    if pw != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin password"}), 403
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file      = request.files["file"]
    filename  = secure_filename(file.filename or "document.txt")
    file_bytes= file.read()
    ext       = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    try:
        if ext == "pdf":    text = extract_pdf(file_bytes)
        elif ext == "docx": text = extract_docx(file_bytes)
        else:               text = extract_txt(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400
    if not text.strip():
        return jsonify({"error": "No readable text found in this file."}), 400
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)
    doc    = {"id": doc_id, "name": filename,
              "uploaded_at": datetime.now().isoformat(),
              "chunk_count": len(chunks), "size": len(text)}
    recs   = [{"id": str(uuid.uuid4()), "doc_id": doc_id, "doc_name": filename,
               "text": c, "index": i} for i, c in enumerate(chunks)]
    knowledge_base["documents"].append(doc)
    knowledge_base["chunks"].extend(recs)
    save_kb(knowledge_base)
    return jsonify({"success": True, "document": doc, "chunks_created": len(chunks)})
 
@app.route("/api/documents")
def get_documents():
    return jsonify(knowledge_base.get("documents", []))
 
@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    global knowledge_base
    pw = (request.json or {}).get("password", "")
    if pw != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin password"}), 403
    knowledge_base["documents"] = [d for d in knowledge_base["documents"] if d["id"] != doc_id]
    knowledge_base["chunks"]    = [c for c in knowledge_base["chunks"]    if c["doc_id"] != doc_id]
    save_kb(knowledge_base)
    return jsonify({"success": True})
 
@app.route("/api/chat", methods=["POST"])
def chat():
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY is not configured."}), 500
    data     = request.json or {}
    msgs     = data.get("messages", [])
    user_msg = msgs[-1]["content"] if msgs else ""
    relevant = search(user_msg, knowledge_base.get("chunks", []), top_k=6)
    if relevant:
        context  = "\n\n---\n\n".join(
            "[Source: {}]\n{}".format(c["doc_name"], c["text"]) for c in relevant)
        ctx_note = "Answer based ONLY on the context above."
    else:
        context  = "No documents have been uploaded to the knowledge base yet."
        ctx_note = "Inform the user that no documents are available yet."
    system_prompt = (
        "You are {}, an expert AI assistant for high school debate coaches and judges.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY using information from the KNOWLEDGE BASE CONTEXT below.\n"
        "2. If the answer is not in the context, say: "
        "\"I don't have that information in my knowledge base. "
        "Please consult your coach or the official rulebook.\"\n"
        "3. NEVER invent facts, rules, or examples not in the context.\n"
        "4. Mention which source document your answer comes from.\n"
        "5. Be concise and clear for coaches and judges.\n\n"
        "KNOWLEDGE BASE CONTEXT:\n{}\n\n{}"
    ).format(BOT_NAME, context, ctx_note)
    try:
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=MODEL, max_tokens=1024, system=system_prompt,
            messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
        )
        return jsonify({
            "response": response.content[0].text,
            "sources":  [c["doc_name"] for c in relevant],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print("\n{}\n  {}\n{}".format("="*50, BOT_NAME, "="*50))
    print("  URL:     http://localhost:{}".format(port))
    print("  API key: {}".format("Configured" if ANTHROPIC_API_KEY else "NOT SET"))
    print("  Admin:   {}\n{}".format(ADMIN_PASSWORD, "="*50))
    app.run(host="0.0.0.0", port=port, debug=debug)
 # ── Knowledge base ────────────────────────────────────────────────────────────
def load_kb():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "chunks": []}

def save_kb(kb):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

knowledge_base = load_kb()

# ── Text extraction ───────────────────────────────────────────────────────────
def extract_pdf(file_bytes):
    if HAS_PDFPLUMBER:
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                parts = []
                text = page.extract_text()
                if text and text.strip():
                    parts.append(text.strip())
                # Extract tables as readable text
                for table in (page.extract_tables() or []):
                    if not table:
                        continue
                    headers = [str(c).strip() if c else "" for c in table[0]]
                    has_headers = any(h and not h.replace(" ","").isnumeric() for h in headers)
                    rows = []
                    for ri, row in enumerate(table):
                        if ri == 0 and has_headers:
                            continue
                        cells = [str(c).strip() if c else "" for c in row]
                        if not any(cells):
                            continue
                        if has_headers:
                            rows.append(" | ".join(
                                f"{h}: {v}" for h, v in zip(headers, cells) if h or v
                            ))
                        else:
                            rows.append(" | ".join(cells))
                    if rows:
                        parts.append("TABLE:\n" + "\n".join(rows))
                if parts:
                    pages.append(f"[Page {i+1}]\n" + "\n\n".join(parts))
        return "\n\n".join(pages)
    elif HAS_FITZ:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {i+1}]\n{text}")
        return "\n\n".join(pages)
    else:
        raise ValueError("No PDF library available. Run: pip install pdfplumber")

def extract_docx(file_bytes):
    if not HAS_DOCX:
        raise ValueError("python-docx not installed")
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="replace")

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=500, overlap=60):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = []
    for para in paragraphs:
        words = para.split()
        if len(current) + len(words) <= chunk_size:
            current.extend(words)
        else:
            if current:
                chunks.append(" ".join(current))
            if len(words) > chunk_size:
                for i in range(0, len(words), chunk_size - overlap):
                    seg = " ".join(words[i:i + chunk_size])
                    if seg:
                        chunks.append(seg)
                current = words[-overlap:]
            else:
                current = words
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.split()) >= 10]

# ── Retrieval ─────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","on","at","by","for","with","from","this",
    "that","it","its","what","how","when","where","who","which","and","or",
    "but","if","then","than","as","so","yet","both","either","neither","not",
    "no","nor","just","because","although","though","while","i","you","he",
    "she","we","they","me","him","her","us","them","about","also","any",
    "each","into","through","during","before","after","above","below","up",
    "down","out","off","over","under","again","further","there","here",
}

DEBATE_SYNONYMS = {
    "counterplan":["counter plan","counter-plan","cp"],
    "counter plan":["counterplan","counter-plan","cp"],
    "cp":["counterplan","counter plan"],
    "disadvantage":["disad","da","disadvantages"],
    "disad":["disadvantage","da"],
    "da":["disadvantage","disad"],
    "kritik":["critique","k","critical argument"],
    "critique":["kritik","k"],
    "k":["kritik","critique"],
    "topicality":["t","on-topic","topical"],
    "t":["topicality"],
    "permutation":["perm","permutations"],
    "perm":["permutation"],
    "solvency":["solve","solves"],
    "framework":["fw","value framework","criterion"],
    "fw":["framework"],
    "evidence":["ev","card","cards"],
    "ev":["evidence","card"],
    "card":["evidence","ev"],
    "affirmative":["aff","pro"],
    "aff":["affirmative","pro"],
    "negative":["neg","con"],
    "neg":["negative","con"],
    "public forum":["pf","pfd"],
    "pf":["public forum","public forum debate"],
    "lincoln douglas":["ld","lincoln-douglas"],
    "ld":["lincoln douglas","lincoln-douglas"],
    "policy debate":["cx","policy"],
    "cx":["policy debate","policy"],
    "congressional":["congress","student congress"],
    "congress":["congressional","student congress"],
    "extemp":["extemporaneous","extemporaneous speaking"],
    "extemporaneous":["extemp"],
    "crossfire":["cross examination","cross-ex"],
    "cross examination":["crossfire","cross-ex","cx"],
    "flow":["flowing","flowed"],
    "rebuttal":["rebuttals","rebut"],
    "contention":["contentions","argument"],
    "speaker points":["speaks","speaker point"],
    "speaks":["speaker points","speaker point"],
    "ballot":["ballots","decision"],
    "resolution":["res","topic"],
    "res":["resolution","topic"],
    "prep time":["prep","preparation time"],
    "prep":["prep time","preparation time"],
}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"(\w)-(\w)", r"\1\2", text)
    return text

def stem_word(word):
    if len(word) <= 4:
        return word
    for suffix in ["ations","ation","ings","ing","tions","tion",
                   "ness","ments","ment","ers","ies","ed","es","s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def get_keywords(text):
    words = re.findall(r"\b[a-z]{2,}\b", normalize_text(text))
    return {w for w in words if w not in STOP_WORDS}

def expand_query(query):
    expanded = set()
    q_lower = query.lower()
    q_norm  = normalize_text(query)
    base = get_keywords(q_lower) | get_keywords(q_norm)
    expanded.update(base)
    expanded.update(stem_word(w) for w in base)
    for word in list(base):
        if word in DEBATE_SYNONYMS:
            for syn in DEBATE_SYNONYMS[word]:
                expanded.update(get_keywords(syn))
                expanded.add(normalize_text(syn).replace(" ", ""))
    for phrase, synonyms in DEBATE_SYNONYMS.items():
        if phrase in q_lower or phrase in q_norm:
            for syn in synonyms:
                expanded.update(get_keywords(syn))
    expanded.add(re.sub(r"\s+", "", q_norm))
    return expanded - STOP_WORDS

def search(query, chunks, top_k=6):
    if not chunks:
        return []
    q_kw = expand_query(query)
    if not q_kw:
        return chunks[:top_k]
    scored = []
    for chunk in chunks:
        raw   = chunk["text"]
        norm  = normalize_text(raw)
        words = re.findall(r"\b[a-z]{2,}\b", norm)
        dlen  = len(words) + 1
        c_kw  = {w for w in words if w not in STOP_WORDS}
        c_all = c_kw | {stem_word(w) for w in c_kw}
        score = 0.0
        for kw in q_kw:
            ks = stem_word(kw)
            if kw in c_all or ks in c_all:
                tf = (words.count(kw) + words.count(ks)) / dlen
                score += (tf + 0.01) * (1 + 1 / max(len(kw) ** 0.3, 1))
        for kw in q_kw:
            if len(kw) > 5 and kw in norm:
                score += 0.2
        q_lower = query.lower()
        for phrase in DEBATE_SYNONYMS:
            if len(phrase) > 4 and phrase in q_lower and phrase in norm:
                score += 0.4
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = (
    "<!DOCTYPE html>"
    "<html lang='en'>"
    "<head>"
    "<meta charset='UTF-8'/>"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'/>"
    "<title>Debate AI Assistant</title>"
    "<style>"
    "*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}"
    ":root{--navy:#1a2744;--navy2:#243460;--gold:#c9a227;--gold2:#e0b93a;"
    "--light:#f4f6fb;--white:#ffffff;--gray:#6b7280;--border:#dde3f0;"
    "--red:#dc2626;--green:#16a34a;--radius:12px;--shadow:0 4px 24px rgba(26,39,68,.12)}"
    "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
    "background:var(--light);color:#1f2937;min-height:100vh;display:flex;flex-direction:column}"
    "header{background:var(--navy);color:var(--white);padding:0 24px;height:64px;"
    "display:flex;align-items:center;justify-content:space-between;"
    "box-shadow:0 2px 12px rgba(0,0,0,.25);position:sticky;top:0;z-index:100}"
    ".header-left{display:flex;align-items:center;gap:12px}"
    ".logo-icon{width:38px;height:38px;background:var(--gold);border-radius:8px;"
    "display:flex;align-items:center;justify-content:center;font-size:20px}"
    ".brand h1{font-size:18px;font-weight:700;letter-spacing:-.3px}"
    ".brand p{font-size:12px;opacity:.7;margin-top:1px}"
    ".header-right{display:flex;align-items:center;gap:10px}"
    ".status-pill{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,.1);"
    "border-radius:20px;padding:4px 12px;font-size:12px}"
    ".status-dot{width:7px;height:7px;border-radius:50%;background:#9ca3af}"
    ".status-dot.online{background:#34d399}"
    ".status-dot.offline{background:var(--red)}"
    ".admin-btn{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);"
    "color:var(--white);border-radius:8px;padding:6px 14px;font-size:13px;cursor:pointer;"
    "display:flex;align-items:center;gap:6px;transition:background .2s}"
    ".admin-btn:hover{background:rgba(255,255,255,.25)}"
    ".main{display:flex;flex:1;gap:0;overflow:hidden}"
    ".admin-panel{width:340px;min-width:340px;background:var(--white);"
    "border-right:1px solid var(--border);display:none;flex-direction:column;overflow:hidden}"
    ".admin-panel.open{display:flex}"
    ".admin-header{background:var(--navy2);color:var(--white);padding:16px 20px;"
    "font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px}"
    ".admin-body{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:20px}"
    ".admin-section-title{font-size:11px;font-weight:700;text-transform:uppercase;"
    "letter-spacing:.8px;color:var(--gray);margin-bottom:8px}"
    ".lock-screen{display:flex;flex-direction:column;gap:10px}"
    ".lock-screen p{font-size:13px;color:var(--gray)}"
    "label{font-size:13px;font-weight:500;display:block;margin-bottom:4px}"
    "input[type=password],input[type=text]{width:100%;padding:9px 12px;"
    "border:1px solid var(--border);border-radius:8px;font-size:14px;outline:none;transition:border .2s}"
    "input:focus{border-color:var(--navy2)}"
    ".btn{padding:9px 18px;border-radius:8px;border:none;font-size:14px;font-weight:600;"
    "cursor:pointer;transition:background .2s,transform .1s}"
    ".btn:active{transform:scale(.97)}"
    ".btn-primary{background:var(--navy);color:var(--white)}"
    ".btn-primary:hover{background:var(--navy2)}"
    ".btn-danger{background:#fee2e2;color:var(--red)}"
    ".btn-danger:hover{background:#fecaca}"
    ".btn-sm{padding:5px 12px;font-size:12px}"
    ".btn-block{width:100%}"
    ".upload-zone{border:2px dashed var(--border);border-radius:var(--radius);"
    "padding:24px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;position:relative}"
    ".upload-zone:hover,.upload-zone.drag{border-color:var(--navy2);background:#f0f4ff}"
    ".upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}"
    ".upload-zone .upload-icon{font-size:32px;margin-bottom:8px}"
    ".upload-zone p{font-size:13px;color:var(--gray)}"
    ".upload-zone strong{color:var(--navy)}"
    ".progress-bar{height:6px;background:var(--border);border-radius:3px;overflow:hidden;display:none;margin-top:8px}"
    ".progress-bar-fill{height:100%;background:var(--gold);border-radius:3px;width:0%;transition:width .3s}"
    ".doc-list{display:flex;flex-direction:column;gap:8px}"
    ".doc-item{background:var(--light);border:1px solid var(--border);border-radius:8px;"
    "padding:10px 12px;display:flex;align-items:flex-start;justify-content:space-between;gap:8px}"
    ".doc-info{flex:1;min-width:0}"
    ".doc-name{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
    ".doc-meta{font-size:11px;color:var(--gray);margin-top:2px}"
    ".doc-icon{font-size:18px;margin-right:6px;flex-shrink:0}"
    ".doc-left{display:flex;align-items:center;flex:1;min-width:0}"
    ".no-docs{font-size:13px;color:var(--gray);text-align:center;padding:16px 0}"
    ".alert{padding:10px 14px;border-radius:8px;font-size:13px;display:none}"
    ".alert.show{display:block}"
    ".alert-success{background:#dcfce7;color:var(--green);border:1px solid #bbf7d0}"
    ".alert-error{background:#fee2e2;color:var(--red);border:1px solid #fecaca}"
    ".chat-area{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}"
    ".chat-messages{flex:1;overflow-y:auto;padding:24px 20px;"
    "display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth}"
    ".welcome-card{max-width:520px;margin:20px auto;background:var(--white);"
    "border-radius:var(--radius);padding:32px;text-align:center;"
    "box-shadow:var(--shadow);border:1px solid var(--border)}"
    ".welcome-card .big-icon{font-size:48px;margin-bottom:12px}"
    ".welcome-card h2{font-size:22px;color:var(--navy);margin-bottom:8px}"
    ".welcome-card p{font-size:14px;color:var(--gray);line-height:1.6}"
    ".welcome-card .tips{margin-top:20px;text-align:left;background:var(--light);"
    "border-radius:8px;padding:14px 16px}"
    ".welcome-card .tips p{font-size:13px;margin-bottom:4px}"
    ".tip-label{font-weight:700;color:var(--navy)}"
    ".message{display:flex;gap:10px;align-items:flex-start;max-width:800px}"
    ".message.user{flex-direction:row-reverse;margin-left:auto}"
    ".message.bot{margin-right:auto}"
    ".avatar{width:34px;height:34px;border-radius:50%;flex-shrink:0;"
    "display:flex;align-items:center;justify-content:center;font-size:16px}"
    ".message.user .avatar{background:var(--navy);color:var(--white);font-size:13px;font-weight:700}"
    ".message.bot .avatar{background:var(--gold);color:var(--navy)}"
    ".bubble{padding:12px 16px;border-radius:var(--radius);font-size:14px;"
    "line-height:1.6;max-width:100%;word-break:break-word}"
    ".message.user .bubble{background:var(--navy);color:var(--white);border-top-right-radius:4px}"
    ".message.bot .bubble{background:var(--white);border:1px solid var(--border);"
    "border-top-left-radius:4px;box-shadow:0 1px 6px rgba(0,0,0,.06)}"
    ".sources-tag{display:flex;align-items:center;gap:6px;margin-top:8px;flex-wrap:wrap}"
    ".source-label{font-size:11px;color:var(--gray);font-weight:600}"
    ".source-chip{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;"
    "border-radius:20px;padding:2px 10px;font-size:11px;white-space:nowrap;"
    "overflow:hidden;text-overflow:ellipsis;max-width:180px}"
    ".typing-bubble{background:var(--white);border:1px solid var(--border);"
    "border-radius:var(--radius);border-top-left-radius:4px;padding:14px 18px;"
    "display:flex;gap:5px;align-items:center;box-shadow:0 1px 6px rgba(0,0,0,.06)}"
    ".dot{width:7px;height:7px;border-radius:50%;background:#9ca3af;animation:bounce 1.2s infinite}"
    ".dot:nth-child(2){animation-delay:.2s}"
    ".dot:nth-child(3){animation-delay:.4s}"
    "@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-8px)}}"
    ".input-bar{background:var(--white);border-top:1px solid var(--border);padding:16px 20px}"
    ".input-wrap{display:flex;gap:10px;align-items:flex-end;background:var(--light);"
    "border:1.5px solid var(--border);border-radius:12px;padding:10px 14px;transition:border-color .2s}"
    ".input-wrap:focus-within{border-color:var(--navy2);background:var(--white)}"
    "#user-input{flex:1;border:none;background:transparent;font-size:14px;line-height:1.5;"
    "resize:none;outline:none;max-height:120px;min-height:24px;font-family:inherit}"
    ".send-btn{background:var(--navy);color:var(--white);border:none;border-radius:8px;"
    "width:36px;height:36px;cursor:pointer;display:flex;align-items:center;justify-content:center;"
    "flex-shrink:0;transition:background .2s;font-size:16px}"
    ".send-btn:hover:not(:disabled){background:var(--navy2)}"
    ".send-btn:disabled{opacity:.4;cursor:not-allowed}"
    ".input-hint{font-size:11px;color:var(--gray);margin-top:6px;text-align:center}"
    ".no-kb-banner{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;"
    "padding:10px 16px;font-size:13px;color:#92400e;margin:0 20px;"
    "display:flex;align-items:center;gap:8px}"
    ".no-kb-banner.hidden{display:none}"
    "@media(max-width:640px){.admin-panel{width:100%;min-width:unset;position:absolute;"
    "z-index:50;height:100%;top:64px;left:0}.brand p{display:none}}"
    "</style>"
    "</head>"
    "<body>"
    "<header>"
    "<div class='header-left'>"
    "<div class='logo-icon'>&#9878;</div>"
    "<div class='brand'>"
    "<h1 id='bot-name-header'>Debate Assistant</h1>"
    "<p>AI Knowledge Base &middot; Coaches &amp; Judges</p>"
    "</div></div>"
    "<div class='header-right'>"
    "<div class='status-pill'>"
    "<div class='status-dot' id='status-dot'></div>"
    "<span id='status-text'>Connecting&hellip;</span>"
    "</div>"
    "<button class='admin-btn' onclick='toggleAdmin()'>&#9881; Admin</button>"
    "</div></header>"
    "<div class='main'>"
    "<aside class='admin-panel' id='admin-panel'>"
    "<div class='admin-header'>&#128274; Admin Panel</div>"
    "<div class='admin-body'>"
    "<div id='lock-screen'>"
    "<div class='admin-section-title'>Authentication</div>"
    "<div class='lock-screen'>"
    "<p>Enter your admin password to manage the knowledge base.</p>"
    "<div><label for='admin-pw'>Password</label>"
    "<input type='password' id='admin-pw' placeholder='Enter password&hellip;'"
    " onkeydown='if(event.key===\"Enter\")verifyAdmin()'/></div>"
    "<button class='btn btn-primary btn-block' onclick='verifyAdmin()'>Unlock &rarr;</button>"
    "<div class='alert alert-error' id='pw-error'>Incorrect password. Try again.</div>"
    "</div></div>"
    "<div id='admin-content' style='display:none;flex-direction:column;gap:20px'>"
    "<div><div class='admin-section-title'>Upload Documents</div>"
    "<div class='upload-zone' id='upload-zone'"
    " ondragover=\"event.preventDefault();this.classList.add('drag')\""
    " ondragleave=\"this.classList.remove('drag')\" ondrop='handleDrop(event)'>"
    "<input type='file' id='file-input' multiple accept='.pdf,.docx,.txt'"
    " onchange='handleFileSelect(event)'/>"
    "<div class='upload-icon'>&#128193;</div>"
    "<p><strong>Click or drag files here</strong></p>"
    "<p>PDF, DOCX, TXT up to 50 MB</p>"
    "</div>"
    "<div class='progress-bar' id='progress-bar'>"
    "<div class='progress-bar-fill' id='progress-fill'></div></div>"
    "<div class='alert alert-success' id='upload-success'></div>"
    "<div class='alert alert-error' id='upload-error'></div>"
    "</div>"
    "<div><div class='admin-section-title'>Knowledge Base (<span id='doc-count'>0</span> docs)</div>"
    "<div class='doc-list' id='doc-list'><p class='no-docs'>No documents uploaded yet.</p></div>"
    "</div>"
    "<button class='btn btn-danger btn-sm btn-block' onclick='signOutAdmin()'>Sign Out</button>"
    "</div></div></aside>"
    "<section class='chat-area'>"
    "<div class='no-kb-banner hidden' id='no-kb-banner'>"
    "&#9888; No documents in the knowledge base yet. Open Admin to upload files."
    "</div>"
    "<div class='chat-messages' id='chat-messages'>"
    "<div class='welcome-card' id='welcome-card'>"
    "<div class='big-icon'>&#9878;</div>"
    "<h2 id='bot-name-welcome'>Debate Assistant</h2>"
    "<p>Your AI-powered resource for high school debate rules, procedures, and best practices."
    " Ask me anything &mdash; I will answer only from your uploaded knowledge base.</p>"
    "<div class='tips'>"
    "<p><span class='tip-label'>Try asking:</span></p>"
    "<p>&bull; What are the speaker point guidelines?</p>"
    "<p>&bull; How does time allocation work in LD?</p>"
    "<p>&bull; What is a topicality argument?</p>"
    "<p>&bull; Can the Aff run a CP?</p>"
    "</div></div></div>"
    "<div class='input-bar'>"
    "<div class='input-wrap'>"
    "<textarea id='user-input' rows='1' placeholder='Ask a debate question&hellip;'"
    " onkeydown='handleKey(event)' oninput='autoResize(this)'></textarea>"
    "<button class='send-btn' id='send-btn' onclick='sendMessage()'>&#9658;</button>"
    "</div>"
    "<div class='input-hint'>Answers come only from your uploaded documents &middot; Press Enter to send</div>"
    "</div></section></div>"
    "<script>"
    "var messages=[];"
    "var adminUnlocked=false;"
    "var adminPW='';"
    "async function init(){"
    "  try{"
    "    var r=await fetch('/api/status');"
    "    var s=await r.json();"
    "    document.getElementById('bot-name-header').textContent=s.bot_name;"
    "    document.getElementById('bot-name-welcome').textContent=s.bot_name;"
    "    var dot=document.getElementById('status-dot');"
    "    var txt=document.getElementById('status-text');"
    "    if(s.has_api_key){dot.className='status-dot online';txt.textContent='Online';}"
    "    else{dot.className='status-dot offline';txt.textContent='No API key';}"
    "    if(s.document_count===0)document.getElementById('no-kb-banner').classList.remove('hidden');"
    "    document.getElementById('doc-count').textContent=s.document_count;"
    "  }catch(e){console.error(e);}"
    "}"
    "function toggleAdmin(){"
    "  var p=document.getElementById('admin-panel');"
    "  p.classList.toggle('open');"
    "  if(p.classList.contains('open')&&adminUnlocked)loadDocuments();"
    "}"
    "async function verifyAdmin(){"
    "  var pw=document.getElementById('admin-pw').value;"
    "  try{"
    "    var r=await fetch('/api/verify-admin',{method:'POST',"
    "      headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});"
    "    var d=await r.json();"
    "    if(d.valid){"
    "      adminPW=pw;adminUnlocked=true;"
    "      document.getElementById('lock-screen').style.display='none';"
    "      document.getElementById('admin-content').style.display='flex';"
    "      loadDocuments();"
    "    }else{showAlert('pw-error',true);}"
    "  }catch(e){showAlert('pw-error',true);}"
    "}"
    "function signOutAdmin(){"
    "  adminUnlocked=false;adminPW='';"
    "  document.getElementById('lock-screen').style.display='block';"
    "  document.getElementById('admin-content').style.display='none';"
    "  document.getElementById('admin-pw').value='';"
    "}"
    "async function loadDocuments(){"
    "  try{"
    "    var r=await fetch('/api/documents');"
    "    var docs=await r.json();"
    "    renderDocList(docs);"
    "    document.getElementById('doc-count').textContent=docs.length;"
    "    var b=document.getElementById('no-kb-banner');"
    "    if(docs.length===0)b.classList.remove('hidden');"
    "    else b.classList.add('hidden');"
    "  }catch(e){console.error(e);}"
    "}"
    "function renderDocList(docs){"
    "  var el=document.getElementById('doc-list');"
    "  if(!docs.length){el.innerHTML='<p class=\"no-docs\">No documents uploaded yet.</p>';return;}"
    "  el.innerHTML=docs.map(function(d){"
    "    var ext=d.name.split('.').pop().toLowerCase();"
    "    var icon=ext==='pdf'?'&#128196;':ext==='docx'?'&#128221;':'&#128203;';"
    "    var date=new Date(d.uploaded_at).toLocaleDateString();"
    "    return '<div class=\"doc-item\"><div class=\"doc-left\"><span class=\"doc-icon\">'+icon+'</span>'"
    "    +'<div class=\"doc-info\"><div class=\"doc-name\" title=\"'+d.name+'\">'+d.name+'</div>'"
    "    +'<div class=\"doc-meta\">'+d.chunk_count+' chunks &middot; '+date+'</div></div></div>'"
    "    +'<button class=\"btn btn-danger btn-sm\" onclick=\"deleteDoc(\\'' +d.id+ '\\',\\'' +d.name+ \\')\">&#10005;</button></div>';"
    "  }).join('');"
    "}"
    "async function deleteDoc(id,name){"
    "  if(!confirm('Remove '+name+' from the knowledge base?'))return;"
    "  try{"
    "    var r=await fetch('/api/documents/'+id,{method:'DELETE',"
    "      headers:{'Content-Type':'application/json'},body:JSON.stringify({password:adminPW})});"
    "    var d=await r.json();"
    "    if(d.success)loadDocuments();"
    "  }catch(e){alert('Delete failed.');}"
    "}"
    "function handleFileSelect(e){uploadFiles(e.target.files);}"
    "function handleDrop(e){"
    "  e.preventDefault();"
    "  document.getElementById('upload-zone').classList.remove('drag');"
    "  uploadFiles(e.dataTransfer.files);"
    "}"
    "async function uploadFiles(fileList){"
    "  for(var i=0;i<fileList.length;i++){await uploadFile(fileList[i]);}"
    "  loadDocuments();"
    "  document.getElementById('file-input').value='';"
    "}"
    "async function uploadFile(file){"
    "  var allowed=['pdf','docx','txt'];"
    "  var ext=file.name.split('.').pop().toLowerCase();"
    "  if(allowed.indexOf(ext)<0){showAlert('upload-error','Unsupported file type. Use PDF, DOCX, or TXT.');return;}"
    "  var bar=document.getElementById('progress-bar');"
    "  var fill=document.getElementById('progress-fill');"
    "  bar.style.display='block';fill.style.width='20%';"
    "  var form=new FormData();"
    "  form.append('file',file,file.name);"
    "  form.append('password',adminPW);"
    "  try{"
    "    fill.style.width='60%';"
    "    var r=await fetch('/api/upload',{method:'POST',body:form});"
    "    fill.style.width='100%';"
    "    var d=await r.json();"
    "    setTimeout(function(){bar.style.display='none';fill.style.width='0%';},800);"
    "    if(d.success)showAlert('upload-success','Uploaded '+file.name+' ('+d.chunks_created+' chunks)');"
    "    else showAlert('upload-error',d.error||'Upload failed.');"
    "  }catch(e){bar.style.display='none';showAlert('upload-error','Upload failed.');}"
    "}"
    "function handleKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}}"
    "function autoResize(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,120)+'px';}"
    "async function sendMessage(){"
    "  var input=document.getElementById('user-input');"
    "  var text=input.value.trim();"
    "  if(!text)return;"
    "  var w=document.getElementById('welcome-card');"
    "  if(w)w.remove();"
    "  input.value='';input.style.height='auto';"
    "  messages.push({role:'user',content:text});"
    "  appendMessage('user',text,[]);"
    "  var btn=document.getElementById('send-btn');"
    "  btn.disabled=true;"
    "  var tid='typing-'+Date.now();"
    "  appendTyping(tid);"
    "  try{"
    "    var r=await fetch('/api/chat',{method:'POST',"
    "      headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:messages})});"
    "    var d=await r.json();"
    "    removeTyping(tid);"
    "    if(d.error)appendMessage('bot','Error: '+d.error,[]);"
    "    else{messages.push({role:'assistant',content:d.response});appendMessage('bot',d.response,d.sources||[]);}"
    "  }catch(e){removeTyping(tid);appendMessage('bot','Could not reach the server.',[]);}"
    "  btn.disabled=false;input.focus();"
    "}"
    "function appendMessage(role,text,sources){"
    "  var c=document.getElementById('chat-messages');"
    "  var div=document.createElement('div');"
    "  div.className='message '+role;"
    "  var av=document.createElement('div');av.className='avatar';"
    "  av.textContent=role==='user'?'You':'\\u2696';"
    "  var bub=document.createElement('div');bub.className='bubble';"
    "  bub.innerHTML=esc(text).replace(/\\n/g,'<br>');"
    "  if(sources&&sources.length){"
    "    var u=[...new Set(sources)];"
    "    var sd=document.createElement('div');sd.className='sources-tag';"
    "    sd.innerHTML='<span class=\"source-label\">Sources:</span>'"
    "    +u.map(function(s){return '<span class=\"source-chip\" title=\"'+s+'\">'+s+'</span>';}).join('');"
    "    bub.appendChild(sd);"
    "  }"
    "  div.appendChild(av);div.appendChild(bub);"
    "  c.appendChild(div);c.scrollTop=c.scrollHeight;"
    "}"
    "function appendTyping(id){"
    "  var c=document.getElementById('chat-messages');"
    "  var div=document.createElement('div');"
    "  div.className='message bot';div.id=id;"
    "  div.innerHTML='<div class=\"avatar\">\\u2696</div>'"
    "  +'<div class=\"typing-bubble\"><div class=\"dot\"></div><div class=\"dot\"></div><div class=\"dot\"></div></div>';"
    "  c.appendChild(div);c.scrollTop=c.scrollHeight;"
    "}"
    "function removeTyping(id){var el=document.getElementById(id);if(el)el.remove();}"
    "function esc(s){"
    "  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');"
    "}"
    "function showAlert(id,msg){"
    "  var el=document.getElementById(id);if(!el)return;"
    "  if(typeof msg==='string')el.textContent=msg;"
    "  el.classList.add('show');"
    "  setTimeout(function(){el.classList.remove('show');},5000);"
    "}"
    "init();"
    "</script>"
    "</body></html>"
)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/status")
def status():
    return jsonify({
        "bot_name":       BOT_NAME,
        "has_api_key":    bool(ANTHROPIC_API_KEY),
        "document_count": len(knowledge_base.get("documents", [])),
        "chunk_count":    len(knowledge_base.get("chunks", [])),
    })

@app.route("/api/verify-admin", methods=["POST"])
def verify_admin():
    pw = (request.json or {}).get("password", "")
    return jsonify({"valid": pw == ADMIN_PASSWORD})

@app.route("/api/upload", methods=["POST"])
def upload():
    global knowledge_base
    pw = request.form.get("password", "")
    if pw != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin password"}), 403
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file      = request.files["file"]
    filename  = secure_filename(file.filename or "document.txt")
    file_bytes= file.read()
    ext       = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    try:
        if ext == "pdf":    text = extract_pdf(file_bytes)
        elif ext == "docx": text = extract_docx(file_bytes)
        else:               text = extract_txt(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400
    if not text.strip():
        return jsonify({"error": "No readable text found in this file."}), 400
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)
    doc    = {"id": doc_id, "name": filename,
              "uploaded_at": datetime.now().isoformat(),
              "chunk_count": len(chunks), "size": len(text)}
    recs   = [{"id": str(uuid.uuid4()), "doc_id": doc_id, "doc_name": filename,
               "text": c, "index": i} for i, c in enumerate(chunks)]
    knowledge_base["documents"].append(doc)
    knowledge_base["chunks"].extend(recs)
    save_kb(knowledge_base)
    return jsonify({"success": True, "document": doc, "chunks_created": len(chunks)})

@app.route("/api/documents")
def get_documents():
    return jsonify(knowledge_base.get("documents", []))

@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    global knowledge_base
    pw = (request.json or {}).get("password", "")
    if pw != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin password"}), 403
    knowledge_base["documents"] = [d for d in knowledge_base["documents"] if d["id"] != doc_id]
    knowledge_base["chunks"]    = [c for c in knowledge_base["chunks"]    if c["doc_id"] != doc_id]
    save_kb(knowledge_base)
    return jsonify({"success": True})

@app.route("/api/chat", methods=["POST"])
def chat():
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY is not configured."}), 500
    data     = request.json or {}
    msgs     = data.get("messages", [])
    user_msg = msgs[-1]["content"] if msgs else ""
    relevant = search(user_msg, knowledge_base.get("chunks", []), top_k=6)
    if relevant:
        context  = "\n\n---\n\n".join(
            "[Source: {}]\n{}".format(c["doc_name"], c["text"]) for c in relevant)
        ctx_note = "Answer based ONLY on the context above."
    else:
        context  = "No documents have been uploaded to the knowledge base yet."
        ctx_note = "Inform the user that no documents are available yet."
    system_prompt = (
        "You are {}, an expert AI assistant for high school debate coaches and judges.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY using information from the KNOWLEDGE BASE CONTEXT below.\n"
        "2. If the answer is not in the context, say: "
        "\"I don't have that information in my knowledge base. "
        "Please consult your coach or the official rulebook.\"\n"
        "3. NEVER invent facts, rules, or examples not in the context.\n"
        "4. Mention which source document your answer comes from.\n"
        "5. Be concise and clear for coaches and judges.\n\n"
        "KNOWLEDGE BASE CONTEXT:\n{}\n\n{}"
    ).format(BOT_NAME, context, ctx_note)
    try:
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=MODEL, max_tokens=1024, system=system_prompt,
            messages=[{"role": m["role"], "content": m["content"]} for m in msgs],
        )
        return jsonify({
            "response": response.content[0].text,
            "sources":  [c["doc_name"] for c in relevant],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print("\n{}\n  {}\n{}".format("="*50, BOT_NAME, "="*50))
    print("  URL:     http://localhost:{}".format(port))
    print("  API key: {}".format("Configured" if ANTHROPIC_API_KEY else "NOT SET"))
    print("  Admin:   {}\n{}".format(ADMIN_PASSWORD, "="*50))
    app.run(host="0.0.0.0", port=port, debug=debug)

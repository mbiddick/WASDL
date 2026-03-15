#!/usr/bin/env python3
import io, json, os, re, uuid
from datetime import datetime
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import anthropic

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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_PASSWORD    = os.environ.get("ADMIN_PASSWORD", "debate2024")
BOT_NAME          = os.environ.get("BOT_NAME", "Debate Assistant")
KB_FILE           = os.environ.get("KB_FILE", "knowledge_base.json")
MODEL             = "claude-haiku-4-5-20251001"

def load_kb():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "chunks": []}

def save_kb(kb):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

knowledge_base = load_kb()

def extract_pdf(file_bytes):
    if not HAS_FITZ:
        raise ValueError("PyMuPDF (fitz) is not installed. Run: pip install pymupdf")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append("[Page {}]\n{}".format(i + 1, text))
    return "\n\n".join(pages)

def extract_docx(file_bytes):
    if not HAS_DOCX:
        raise ValueError("python-docx not installed")
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="replace")

MAX_CHUNKS_PER_DOC = 60  # hard cap per document to keep RAM in check

def chunk_text(text, chunk_size=300, overlap=40):
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
    chunks = [c for c in chunks if len(c.split()) >= 10]
    # Cap chunks per document — evenly spaced to preserve coverage
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        step = len(chunks) / MAX_CHUNKS_PER_DOC
        chunks = [chunks[int(i * step)] for i in range(MAX_CHUNKS_PER_DOC)]
    return chunks

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
    "cross examination":["crossfire","cross-ex"],
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
    # PF speech names
    "final focus":["ff","final speech","final focuses"],
    "ff":["final focus","final speech"],
    "summary":["summary speech","sum","second summary","first summary"],
    "summary speech":["summary","pf summary","sum speech"],
    "grand crossfire":["grand cross","gcx"],
    "gcx":["grand crossfire","grand cross"],
    "second rebuttal":["2nr","second neg rebuttal","final rebuttal"],
    "first summary":["1s","opening summary"],
    "second summary":["2s","closing summary"],
    # LD speech names
    "affirmative constructive":["1ac","ac","aff constructive"],
    "1ac":["affirmative constructive","ac"],
    "negative constructive":["nc","1nc","neg constructive"],
    "nc":["negative constructive","1nc"],
    "negative rebuttal":["nr","2nr"],
    "affirmative rebuttal":["ar","2ar","1ar"],
    "2ar":["affirmative rebuttal","final rebuttal"],
    # Policy speech names
    "1nc":["first negative constructive","negative constructive"],
    "2nc":["second negative constructive"],
    "1ac":["first affirmative constructive","affirmative constructive"],
    "2ac":["second affirmative constructive"],
    "2nr":["second negative rebuttal","negative rebuttal"],
    "2ar":["second affirmative rebuttal","affirmative rebuttal"],
    # General
    "new argument":["new args","new arguments","new constructive"],
    "final speech":["final focus","last speech","closing speech"],
    "constructive":["constructive speech","constructives"],
    "extension":["extend","extending","extensions"],
    "forfeit":["forfeiture","default","no show"],
    "late":["tardy","absence","missing"],
    "disqualification":["dq","disqualify","disqualified"],
    "judge":["judges","judging","adjudicator"],
    "feedback":["oral critique","critique","comments","verbal feedback"],
    "pairing":["pairings","bracket","matchup"],
    "bye":["automatic win","free round"],
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
        norm  = normalize_text(chunk["text"])
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

def expand_query_with_claude(query):
    """Use Claude to generate related search terms for better keyword retrieval."""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=MODEL, max_tokens=120,
            messages=[{"role": "user", "content": (
                "You are helping search a debate rulebook. "
                "Generate 8 related search terms for this query: \"{}\"\n"
                "Include debate-specific synonyms, related rule concepts, and alternate "
                "phrasings that might appear in a rulebook. "
                "Return ONLY a comma-separated list of short terms, nothing else."
            ).format(query)}]
        )
        raw = response.content[0].text.strip()
        return [t.strip() for t in raw.split(",") if t.strip()]
    except Exception:
        return []

def rerank_chunks(query, chunks, top_k=7):
    """Use Claude to semantically rerank candidate chunks by relevance to the query."""
    if not chunks or len(chunks) <= top_k:
        return chunks
    numbered = "\n\n".join(
        "[{}] {}: {}".format(i, c["doc_name"], c["text"][:400])
        for i, c in enumerate(chunks)
    )
    prompt = (
        "A user asked: \"{}\"\n\n"
        "Below are numbered text chunks from a debate rulebook knowledge base. "
        "Your job is to select the chunks that would best help answer the question.\n\n"
        "SELECTION RULES:\n"
        "1. Prioritize chunks that DIRECTLY ANSWER the question — a chunk that states "
        "the rule or gives the specific answer scores higher than one that merely mentions "
        "the same topic in passing.\n"
        "2. Match on MEANING, not just keywords — e.g. 'record a speech' and "
        "'audio/video recording during a round' are the same topic.\n"
        "3. A guidance or overview document may contain the actual rule even if a "
        "format-specific document with the same format name is already selected — "
        "include BOTH if each contributes something to the answer.\n"
        "4. If chunks from different documents address the same question differently "
        "(e.g. one format allows something, another prohibits it), include chunks from "
        "ALL those documents.\n"
        "5. Do NOT prefer a chunk just because its filename matches the format in the "
        "question — prefer the chunk that actually states the rule.\n\n"
        "Return ONLY a comma-separated list of index numbers (e.g. 0, 2, 4). "
        "Select up to {} chunks. If none directly help answer the question, return 'none'.\n\n{}"
    ).format(query, top_k, numbered)
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=MODEL, max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.lower() == "none":
            return []
        indices = [int(x.strip()) for x in re.findall(r"\d+", raw)]
        indices = [i for i in indices if 0 <= i < len(chunks)][:top_k]
        return [chunks[i] for i in indices]
    except Exception:
        return chunks[:top_k]

@app.route("/")
def index():
    return render_template("index.html", bot_name=BOT_NAME)

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
    file       = request.files["file"]
    filename   = secure_filename(file.filename or "document.txt")
    file_bytes = file.read()
    ext        = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    try:
        if ext == "pdf":
            text = extract_pdf(file_bytes)
        elif ext == "docx":
            text = extract_docx(file_bytes)
        else:
            text = extract_txt(file_bytes)
    except Exception as e:
        return jsonify({"error": "Could not parse file: {}".format(e)}), 400
    if not text.strip():
        return jsonify({"error": "No readable text found in this file."}), 400
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)
    doc    = {
        "id": doc_id, "name": filename,
        "uploaded_at": datetime.now().isoformat(),
        "chunk_count": len(chunks), "size": len(text)
    }
    recs = [
        {"id": str(uuid.uuid4()), "doc_id": doc_id, "doc_name": filename,
         "text": c, "index": i}
        for i, c in enumerate(chunks)
    ]
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
    user_msg   = msgs[-1]["content"] if msgs else ""
    all_chunks = knowledge_base.get("chunks", [])

    # Step 1: expand query using Claude to generate related search terms
    expanded_terms = expand_query_with_claude(user_msg)

    # Step 2: global keyword search with original query + each expanded term
    candidates = search(user_msg, all_chunks, top_k=15)
    seen_ids   = {c["id"] for c in candidates}
    for term in expanded_terms:
        for c in search(term, all_chunks, top_k=5):
            if c["id"] not in seen_ids:
                candidates.append(c)
                seen_ids.add(c["id"])

    # Step 3: for EVERY document, guarantee its top-matching chunks are in the pool.
    # Search with both the original query AND expanded terms so vocabulary mismatches
    # (e.g. "summary speech" vs "summary") are covered.
    # Also guarantees every document contributes at least its #1 chunk — this ensures
    # guidance/overview docs are never silently excluded before the reranker runs.
    all_doc_ids = {c["doc_id"] for c in all_chunks}
    for doc_id in all_doc_ids:
        doc_chunks = [c for c in all_chunks if c["doc_id"] == doc_id]
        top_for_doc = search(user_msg, doc_chunks, top_k=4)
        seen_for_doc = {c["id"] for c in top_for_doc}
        # Also search each expanded term against this document
        for term in expanded_terms[:3]:
            for c in search(term, doc_chunks, top_k=2):
                if c["id"] not in seen_for_doc:
                    top_for_doc.append(c)
                    seen_for_doc.add(c["id"])
        # Safety net: if keyword search scored nothing for this doc, still include
        # its very first chunk so the reranker at least sees something from it.
        if not top_for_doc and doc_chunks:
            top_for_doc = [doc_chunks[0]]
        for c in top_for_doc:
            if c["id"] not in seen_ids:
                candidates.append(c)
                seen_ids.add(c["id"])

    relevant = rerank_chunks(user_msg, candidates, top_k=9)

    # Format-pinning — for each format mentioned in the query, guarantee one chunk
    # from EVERY document whose name matches that format.  This prevents a situation
    # where doc A (e.g. judge_instructions) satisfies the "PF covered" check while
    # doc B (e.g. PF_and_LD_GUIDANCE) — which may actually contain the answer — is
    # silently dropped by the reranker.
    FORMAT_KEYWORDS = {
        "policy":        ["policy"],
        "public forum":  ["public forum", "pf"],
        "pf":            ["public forum", "pf"],
        "lincoln":       ["lincoln", "ld"],
        "ld":            ["lincoln", "ld"],
        "congressional": ["congressional", "congress"],
        "congress":      ["congressional", "congress"],
        "speech":        ["speech"],
        "tabroom":       ["tabroom"],
    }
    q_lower = user_msg.lower()
    pinned_formats = [kws for trigger, kws in FORMAT_KEYWORDS.items() if trigger in q_lower]
    if pinned_formats:
        relevant_ids = {c["id"] for c in relevant}
        for fmt_keywords in pinned_formats:
            # Find every document whose name matches this format
            fmt_doc_ids = {
                c["doc_id"] for c in all_chunks
                if any(kw in c["doc_name"].lower() for kw in fmt_keywords)
            }
            # Find which of those docs are already represented in relevant
            covered_doc_ids = {
                c["doc_id"] for c in relevant
                if any(kw in c["doc_name"].lower() for kw in fmt_keywords)
            }
            # For each matching doc not yet represented, add its best candidate chunk
            for doc_id in fmt_doc_ids - covered_doc_ids:
                for c in candidates:
                    if c["id"] not in relevant_ids and c["doc_id"] == doc_id:
                        relevant.append(c)
                        relevant_ids.add(c["id"])
                        break

    if relevant:
        context  = "\n\n---\n\n".join(
            "[Source: {}]\n{}".format(c["doc_name"], c["text"]) for c in relevant
        )
        ctx_note = "Answer based ONLY on the context above."
    else:
        context  = "No documents have been uploaded yet."
        ctx_note = "Tell the user no documents are available yet."
    system_prompt = (
        "You are {name}, an expert AI assistant for high school debate coaches and judges.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY using the KNOWLEDGE BASE CONTEXT below.\n"
        "2. If the answer is not in the context, say exactly: "
        "\"I don't have that information in my knowledge base. "
        "Please consult your coach or the official rulebook.\"\n"
        "3. NEVER invent facts, rules, or examples not in the context.\n"
        "4. Mention which source document your answer comes from.\n"
        "5. Be concise and clear for coaches and judges.\n"
        "6. DEBATE FORMAT RULE: Debate rules often differ by format (Policy, Public Forum, "
        "Lincoln-Douglas, Congressional, etc.). If the user's question does not specify a "
        "format, check the context for each format and address them separately. "
        "Never give an answer that applies to only one format without noting that other "
        "formats may differ. If the context only covers some formats, say so.\n"
        "7. GUIDANCE DOCUMENTS: The knowledge base includes both format-specific rule documents "
        "AND general guidance/overview documents (e.g. PF_and_LD_GUIDANCE, judge_instructions). "
        "Guidance documents often contain the actual rule being asked about. "
        "ALWAYS scan ALL provided context chunks before concluding an answer is not available — "
        "the answer may be in a guidance document rather than a format-named rulebook.\n"
        "8. NEGATIVE RULES COUNT: If the context states that something is NOT permitted, "
        "NOT used, or does NOT apply in a given format, that IS a complete answer — "
        "state it clearly rather than saying the information is unavailable.\n\n"
        "KNOWLEDGE BASE CONTEXT:\n{context}\n\n{note}"
    ).format(name=BOT_NAME, context=context, note=ctx_note)
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
    port = int(os.environ.get("PORT", 5000))
    print("Starting {} on port {}".format(BOT_NAME, port))
    app.run(host="0.0.0.0", port=port)

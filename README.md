# Debate AI Assistant

A RAG-powered chatbot for high school debate coaches and judges.
Answers questions **only** from your uploaded documents — no hallucinations.

---

## Quick Start (run locally)

### 1. Install Python 3.10+
Download from https://python.org if needed.

### 2. Install dependencies
```bash
cd debate-ai-app
pip install -r requirements.txt
```

### 3. Set your Anthropic API key
Get one at https://console.anthropic.com

**Mac / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ADMIN_PASSWORD="yourpassword"   # optional, default: debate2024
export BOT_NAME="Debate Assistant"    # optional, renames the bot
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
set ADMIN_PASSWORD=yourpassword
```

### 4. Run the app
```bash
python app.py
```

Open http://localhost:5000 in your browser.

---

## Uploading Your Knowledge Base

1. Click the **⚙️ Admin** button in the top-right corner
2. Enter your admin password (default: `debate2024`)
3. Drag & drop or click to upload **PDF**, **DOCX**, or **TXT** files
   - Rulebooks, judging guides, flow templates, event overviews, etc.
4. Documents are stored in `knowledge_base.json` and persist between restarts

---

## Deploying to the Web (free, ~5 min)

### Option A: Render.com (recommended)
1. Push this folder to a GitHub repository
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set these environment variables in Render's dashboard:
   - `ANTHROPIC_API_KEY` = your key
   - `ADMIN_PASSWORD` = a strong password
   - `BOT_NAME` = your chatbot's name
5. Set Start Command to: `python app.py`
6. Deploy — you'll get a public URL like `https://your-app.onrender.com`

### Option B: Railway.app
1. Push to GitHub
2. New project → Deploy from GitHub at https://railway.app
3. Add the same environment variables
4. Done!

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `ADMIN_PASSWORD` | `debate2024` | Password for the admin upload panel |
| `BOT_NAME` | `Debate Assistant` | Display name for the chatbot |
| `KB_FILE` | `knowledge_base.json` | Where the knowledge base is stored |
| `PORT` | `5000` | HTTP port |

---

## How It Works (RAG)

1. **Upload** — Documents are parsed and split into ~500-word chunks
2. **Search** — When a user asks a question, the app finds the most relevant chunks using keyword search
3. **Answer** — The top chunks are sent to Claude with strict instructions to answer **only from that context**
4. **No hallucinations** — If the answer isn't in your documents, Claude says so

---

## Customising the Welcome Screen

Edit the placeholder example questions in `app.py` inside the `HTML` variable — search for `Try asking:`.

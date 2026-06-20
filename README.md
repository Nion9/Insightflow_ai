---
title: InsightFlow AI
emoji: 🎥
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Turn any YouTube video into a searchable, askable knowledge base.
---

# 🎥 InsightFlow AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Drop a YouTube link → get a transcript and an LLM that answers questions grounded in it.

InsightFlow extracts audio from a YouTube video with `yt-dlp` + `ffmpeg`, transcribes it locally with OpenAI's Whisper, indexes the transcript in Chroma using sentence-transformers embeddings, and answers questions with Gemini over the retrieved chunks. Sources are shown for every answer.

## How it works

```
YouTube URL
   │
   ▼ yt-dlp + ffmpeg
audio.mp3
   │
   ▼ Whisper (local)
transcript
   │
   ▼ RecursiveCharacterTextSplitter
chunks ──► HuggingFace embeddings ──► Chroma (in-memory)
                                             │
user question ──► retriever (top-k) ─────────┤
                                             ▼
                                       Gemini 2.0 Flash
                                             │
                                             ▼
                                      answer + sources
```

If no Gemini key is configured, the app falls back to returning the best-matching passage instead of failing.

## Quickstart

### Prerequisites
- Python 3.10+
- `ffmpeg` on your PATH
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: download from [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and add `bin/` to PATH
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (optional but recommended)

### Install and run

```bash
git clone https://github.com/Nion9/insightflow-ai.git
cd insightflow-ai

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml and paste your Gemini key

streamlit run app.py
```

Then open <http://localhost:8501>, paste a YouTube URL, hit **Analyze video**, and ask questions.

## Deployment

### Hugging Face Spaces (recommended)

Whisper + torch + Chroma is heavier than Streamlit Cloud's 1 GB free tier really likes, so HF Spaces is the smoother host.

1. Create a new Space at <https://huggingface.co/new-space>, pick **Streamlit** as the SDK.
2. `git push` this repo to the Space's remote. The YAML block at the top of this README is the Space's config.
3. In **Settings → Variables and secrets**, add `GOOGLE_API_KEY`.
4. `packages.txt` installs `ffmpeg` automatically; `requirements.txt` does the Python deps.

### Streamlit Community Cloud

1. Push to GitHub.
2. Create an app at <https://share.streamlit.io>, point it at `app.py`.
3. Under **Settings → Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "..."
   ```
4. `packages.txt` already declares `ffmpeg`.

The `tiny` Whisper model is the safest choice on a 1 GB host.

## Project layout

```
insightflow-ai/
├── app.py                  # Streamlit UI
├── processor.py            # yt-dlp + Whisper
├── brain.py                # Chroma index + Gemini RAG chain
├── requirements.txt        # Pinned Python deps
├── packages.txt            # System deps for HF/Streamlit Cloud (ffmpeg)
├── .streamlit/
│   └── secrets.toml.example
├── .gitignore
├── LICENSE
└── README.md
```

## Configuration

| Setting | Where | Notes |
| --- | --- | --- |
| `GOOGLE_API_KEY` | `.streamlit/secrets.toml` or env | Enables LLM-powered answers. |
| Whisper model | Sidebar (`tiny` / `base` / `small`) | Trade speed for accuracy. |
| Chunk size / overlap | `brain.py` (`build_vectorstore`) | Defaults: 1000 / 150. |
| Retriever `k` | `app.py` and `brain.py` | Defaults to 4. |

## Tech stack

Streamlit · yt-dlp · OpenAI Whisper · LangChain · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · Google Gemini 2.0 Flash.

## Author

**Minhajul Islam Nion**
[GitHub @Nion9](https://github.com/Nion9) · [LinkedIn](https://www.linkedin.com/in/nion007) · minhajulislamnion@gmail.com

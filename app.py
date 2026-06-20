"""InsightFlow AI — Streamlit interface.

Pipeline: YouTube URL → audio → Whisper transcript → Chroma index → RAG chat.
"""
from __future__ import annotations

import os

import streamlit as st

from brain import build_vectorstore, make_qa_chain, search
from processor import cleanup, download_audio, transcribe


st.set_page_config(
    page_title="InsightFlow AI",
    page_icon="🎥",
    layout="wide",
)


# ---------- Helpers ----------

def get_api_key() -> str | None:
    """Look for the Gemini API key in Streamlit secrets, then env."""
    try:
        key = st.secrets["GOOGLE_API_KEY"]
        if key:
            return key
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return os.getenv("GOOGLE_API_KEY")


# ---------- Session state ----------

ss = st.session_state
ss.setdefault("vectorstore", None)
ss.setdefault("transcript", None)
ss.setdefault("chain", None)
ss.setdefault("messages", [])
ss.setdefault("video_url", "")


# ---------- Sidebar ----------

with st.sidebar:
    st.header("⚙️ Settings")

    model_size = st.selectbox(
        "Whisper model",
        ["tiny", "base", "small"],
        index=1,
        help="Larger = more accurate, slower, more memory.",
    )

    st.markdown("---")
    st.markdown("### 🔑 LLM access")
    if get_api_key():
        st.success("Gemini API key detected — full RAG enabled.")
    else:
        st.warning("No `GOOGLE_API_KEY` — Q&A will fall back to passage retrieval.")
        st.caption("Set it in `.streamlit/secrets.toml` or as an env var.")

    st.markdown("---")
    if st.button("🔄 Reset session", use_container_width=True):
        for k in ("vectorstore", "transcript", "chain"):
            ss[k] = None
        ss["messages"] = []
        ss["video_url"] = ""
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Paste a YouTube URL\n"
        "2. Audio is extracted with `yt-dlp` + `ffmpeg`\n"
        "3. Whisper transcribes it\n"
        "4. Text is chunked, embedded, and stored in Chroma\n"
        "5. Gemini answers your questions over the retrieved chunks"
    )


# ---------- Header ----------

st.title("🎥 InsightFlow AI")
st.caption("Turn any YouTube video into a searchable, askable knowledge base.")


# ---------- Step 1: Ingest ----------

url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    value=ss["video_url"],
)

if st.button("🚀 Analyze video", type="primary", disabled=not url):
    audio_path = None
    try:
        with st.status("Processing video…", expanded=True) as status:
            st.write("📥 Downloading audio…")
            audio_path = download_audio(url)

            st.write(f"🎙️ Transcribing with Whisper ({model_size})…")
            transcript = transcribe(audio_path, model_size=model_size)
            if not transcript:
                raise ValueError("Transcription produced no text.")
            ss["transcript"] = transcript

            st.write("🧠 Building vector index…")
            ss["vectorstore"] = build_vectorstore(transcript)
            ss["chain"] = make_qa_chain(ss["vectorstore"], api_key=get_api_key())

            ss["video_url"] = url
            ss["messages"] = []

            status.update(label="✅ Ready", state="complete", expanded=False)
        st.success("Ask anything about the video below.")
    except Exception as e:
        st.error(f"❌ {e}")
    finally:
        if audio_path:
            cleanup(audio_path)


# ---------- Step 2: Transcript ----------

if ss["transcript"]:
    with st.expander("📄 Full transcript", expanded=False):
        st.text_area(
            "transcript",
            ss["transcript"],
            height=300,
            label_visibility="collapsed",
        )
        st.download_button(
            "⬇️ Download transcript",
            ss["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )


# ---------- Step 3: Chat ----------

if ss["vectorstore"]:
    st.divider()
    st.subheader("💬 Ask the video")

    # Replay history
    for msg in ss["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}]**  {src}")

    # New turn
    if prompt := st.chat_input("What's this video about?"):
        ss["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            sources = search(ss["vectorstore"], prompt, k=4)

            if ss["chain"]:
                with st.spinner("Thinking…"):
                    try:
                        answer = ss["chain"].invoke(prompt)
                    except Exception as e:
                        answer = (
                            f"⚠️ LLM error: `{e}`\n\n"
                            f"Best matching passage:\n\n> {sources[0]}"
                        )
            else:
                answer = (
                    "_No LLM connected — returning the best-matching passage._\n\n"
                    f"> {sources[0]}"
                )

            st.markdown(answer)
            with st.expander("📚 Sources"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**[{i}]**  {src}")

            ss["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })


# ---------- Footer ----------

st.markdown("---")
st.caption(
    "Built with Streamlit · Whisper · ChromaDB · LangChain · Gemini  ·  "
    "[github.com/Nion9](https://github.com/Nion9)"
)

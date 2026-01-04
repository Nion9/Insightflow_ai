# 🎥 InsightFlow AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Transform YouTube videos into searchable knowledge with AI-powered transcription and semantic search

InsightFlow AI is an intelligent video processing system that extracts audio from YouTube videos, transcribes content using OpenAI's Whisper, and creates a searchable question-answering system using LangChain's RAG (Retrieval-Augmented Generation) capabilities.

![InsightFlow AI Demo](demo.gif) <!-- Add a demo gif if available -->

## ✨ Features

- 🎯 **AI-Powered Transcription**: Automatically transcribe YouTube videos using OpenAI's Whisper model
- 🧠 **Semantic Search**: Query video content using natural language questions
- 💬 **Interactive Q&A**: Ask specific questions and get relevant answers from the video content
- 📊 **Full Transcript Access**: View and download complete transcriptions
- 🔍 **Vector Database**: Leverages ChromaDB for efficient semantic search
- 🚀 **Modern UI**: Clean, responsive interface built with Streamlit

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Transcription**: OpenAI Whisper
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM Framework**: LangChain
- **Video Processing**: yt-dlp, FFmpeg

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- FFmpeg ([Download here](https://ffmpeg.org/download.html))
- pip (Python package manager)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/insightflow-ai.git
cd insightflow-ai
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

**Windows:**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to a directory (e.g., `D:\ffmpeg\bin`)
3. Add to System PATH or update the path in `processor.py`

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using InsightFlow AI

1. **Paste YouTube URL**: Enter the URL of the YouTube video you want to analyze
2. **Click "Analyze Video"**: Wait for the processing to complete (1-3 minutes depending on video length)
3. **Ask Questions**: Once processing is complete, ask questions about the video content
4. **View Transcript**: Expand the transcript section to see the full text

### Example Questions

- "What is the main topic of this video?"
- "Can you summarize the key points discussed?"
- "What does the speaker say about [specific topic]?"
- "What are the recommendations mentioned?"

## 📁 Project Structure

```
insightflow-ai/
│
├── app.py                 # Main Streamlit application
├── processor.py           # Video download and transcription logic
├── brain.py              # Vector database and RAG implementation
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
│
├── .vscode/
│   └── launch.json      # VS Code debug configuration
│
├── venv/                # Virtual environment (not tracked)
├── chroma_db/           # Vector database storage (generated)
└── temp_audio.mp3       # Temporary audio files (generated)
```

## 🔧 Configuration

### FFmpeg Path (Windows Users)

If FFmpeg is not in your system PATH, update the path in `processor.py`:

```python
os.environ["PATH"] += os.pathsep + r"YOUR_FFMPEG_PATH\bin"
```

### Whisper Model Selection

You can change the Whisper model for different accuracy/speed tradeoffs in `processor.py`:

```python
# Options: tiny, base, small, medium, large
model = whisper.load_model("base")  # Change "base" to your preferred model
```

| Model  | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| tiny   | ⚡⚡⚡ | ⭐⭐    | Quick testing |
| base   | ⚡⚡  | ⭐⭐⭐  | Default, balanced |
| small  | ⚡    | ⭐⭐⭐⭐ | Better accuracy |
| medium | 🐌   | ⭐⭐⭐⭐⭐ | High accuracy |
| large  | 🐌🐌 | ⭐⭐⭐⭐⭐ | Best accuracy |

## 🧪 Development

### Running in Debug Mode (VS Code)

1. Open `app.py` in VS Code
2. Press `F5` or click "Run and Debug"
3. Select "Python: Streamlit" configuration

### Testing Individual Components

**Test Processor:**
```bash
python processor.py
```

**Test Brain (Vector DB):**
```bash
python brain.py
```

## 📦 Dependencies

```txt
yt-dlp                    # YouTube video downloader
openai-whisper           # Audio transcription
langchain-text-splitters # Text chunking
langchain-community      # LangChain integrations
langchain-core           # LangChain core functionality
chromadb                 # Vector database
sentence-transformers    # Text embeddings
torch                    # PyTorch for ML models
streamlit                # Web interface
```

## 🎯 Use Cases

- 📚 **Educational Content**: Extract key information from lectures and tutorials
- 🎙️ **Podcast Analysis**: Search through podcast episodes for specific topics
- 📺 **Video Research**: Quickly find relevant sections in long-form content
- 📝 **Meeting Recordings**: Create searchable transcripts of recorded meetings
- 🎬 **Content Creation**: Analyze competitor videos or research topics

## 🛣️ Roadmap

- [ ] Support for multiple video sources (Vimeo, local files)
- [ ] Multi-language support
- [ ] Export functionality (PDF, DOCX)
- [ ] Timestamp-based search results
- [ ] Video player integration with auto-jump to relevant sections
- [ ] Batch processing for multiple videos
- [ ] Advanced analytics dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the incredible speech recognition model
- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Streamlit](https://streamlit.io/) for the easy-to-use web framework
- [ChromaDB](https://www.trychroma.com/) for the vector database

## 👤 Author

**Minhajul Islam Nion**
- Email: minhajulislamnion@gmail.com
- University: University of Canberra
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

## 📧 Contact

For questions or feedback, please reach out via email or open an issue on GitHub.

---

<div align="center">
Made with ❤️ by Nion | Built for recruiters and AI enthusiasts
</div>

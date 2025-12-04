# FITTR AI Assistant - RAG Chatbot

AI-powered conversational assistant for health, fitness, and nutrition using Retrieval-Augmented Generation (RAG).

## 🚀 Quick Deploy to Streamlit Cloud

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

### Prerequisites
- GitHub account
- OpenAI API key

### Deployment Steps

1. **Fork/Clone this repository**
2. **Go to [Streamlit Cloud](https://share.streamlit.io)**
3. **Connect your GitHub repository**
4. **Set the main file:** `streamlit_app.py`
5. **Add secrets** in Streamlit Cloud dashboard:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```
6. **Click Deploy!**

Your app will be live at: `https://your-username-fittr-chatbot.streamlit.app`

## 🛠️ Local Development

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Chatbot_using_rag.git
cd Chatbot_using_rag

# Create virtual environment
python -m venv ragenv_py312
source ragenv_py312/bin/activate  # On Windows: ragenv_py312\Scripts\activate

# Install dependencies
pip install -r requirements_compatible.txt


```

### Run Locally

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## 📊 Features

- ✅ **Advanced RAG Pipeline** with HyDE and MMR
- ✅ **754 Documents** - Research papers + blog articles
- ✅ **LlamaIndex** vector search
- ✅ **OpenAI GPT-4o-mini** for answers
- ✅ **ChromaDB** vector storage
- ✅ **Structured logging** for production monitoring
- ✅ **Conversation memory** for context-aware responses

## 📁 Project Structure

```
Chatbot_using_rag/
├── streamlit_app.py          # Main Streamlit UI
├── chatbot_adv.py            # RAG engine
├── ingest.py                 # Data ingestion
├── utils/
│   └── logger.py             # Structured logging
├── chroma_llamaindex_db/     # Vector database
├── data/                     # Source data
├── requirements.txt          # Dependencies
└── .env.example              # Environment template
```

## 🔐 Environment Variables

Required in `.env` or Streamlit Cloud secrets:

```bash
OPENAI_API_KEY=sk-...        # Required
ENVIRONMENT=production        # Optional (default: development)
```

## 📚 Documentation

- [Installation Guide](INSTALLATION_GUIDE.md)
- [Project Documentation](PROJECT_DOCUMENTATION.md)
- [Quick Reference](QUICK_REFERENCE.md)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙋 Support

For issues or questions, please open a GitHub issue.

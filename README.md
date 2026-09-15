# Mental Health AI Agent

A supportive and safe space powered by AI, designed to provide mental health support, conversation, and guidance through an interactive chatbot interface.

**Live app:** https://mentalhealthaiagent-mbmrndrirhxqgptaav54hb.streamlit.app/

## Features

- Conversational AI therapist powered by advanced language models
- Chat history tracking with user and AI message counts
- Support for uploading and referencing PDF books (for RAG capabilities)
- Safety message with crisis helplines to encourage responsible use
- Easy-to-use Streamlit web interface

## Getting Started

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Lord-LLM/mental_health_AI_agent.git
   cd mental_health_AI_agent
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up any required environment variables (e.g. API keys for the language model provider) in a `.env` file at the project root.

### Running the App

Start the Streamlit app locally:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. Open the app in your browser.
2. Type a message in the chat input to start a conversation.
3. Optionally upload a PDF to have the assistant reference its content in responses.
4. Sidebar stats show the number of messages exchanged in the current session.

## Disclaimer

This app is an AI-powered conversational tool and is not a substitute for professional mental health care. If you or someone you know is in crisis, please reach out to a licensed professional or a crisis helpline immediately.

## License

Add license information here (e.g. MIT, Apache 2.0).

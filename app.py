import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import streamlit as st
from therapist_ai import run_chat
from crisis import SAFETY_MESSAGE

# Page configuration
st.set_page_config(
    page_title="Mental Health AI Agent",
    page_icon=":speech_balloon:",
    layout="wide",
)

# Minimal styling
st.markdown(
    """
    <style>
        .app-title {
            text-align: center;
            font-size: 2rem;
            font-weight: 600;
            color: #4B8BBE;
            margin-bottom: 0.25rem;
        }
        .app-subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        .stat-box {
            background-color: rgba(75, 139, 190, 0.08);
            border-radius: 8px;
            padding: 0.75rem;
            margin-top: 0.5rem;
        }
    </style>
    <div class="app-title">Mental Health AI Agent</div>
    <div class="app-subtitle">A supportive, private space to talk things through.</div>
    """,
    unsafe_allow_html=True,
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.subheader("About")
    st.write(
        "This assistant uses a language model to offer supportive conversation "
        "and general guidance. It is not a substitute for professional care."
    )
    st.markdown(SAFETY_MESSAGE)

    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    ai_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

    st.markdown(
        f"""
        <div class="stat-box">
            <strong>Messages:</strong> {user_msgs + ai_msgs}<br>
            You: {user_msgs} &nbsp;&middot;&nbsp; Assistant: {ai_msgs}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Chat history
st.subheader("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("What's on your mind?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                ai_response = run_chat(user_input)
            except Exception as e:
                ai_response = "Sorry, something went wrong on my end. Please try again in a moment."
                st.error(f"Error: {e}")
        st.markdown(ai_response)

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
import streamlit as st
from therapist_ai import run_chat
from crisis import SAFETY_MESSAGE

# Page Configuration 
st.set_page_config(
    page_title="Mental Health AI Agent 🧠",
    page_icon="🧠",
    layout="wide",
)

# Header 
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 Mental Health AI Agent</h1>
    <p style='text-align: center;'>A supportive and safe space powered by AI — built with ❤️ for wellness.</p>
    """,
    unsafe_allow_html=True
)

# Initialize Session State 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar 
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("This AI agent uses advanced language models to offer mental health support, conversation, and guidance.")
    st.markdown(SAFETY_MESSAGE)

    # Chat tracker
    user_msgs = sum(1 for m in st.session_state["messages"] if m["role"] == "user")
    ai_msgs = sum(1 for m in st.session_state["messages"] if m["role"] == "ai")
    st.markdown(f"**💬 Messages:** {user_msgs + ai_msgs}")
    st.markdown(f"- You: {user_msgs}")
    st.markdown(f"- AI: {ai_msgs}")

# Create a placeholder for the chat display
chat_placeholder = st.empty()

# Chat Input 
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="user_input", placeholder="What's on your mind?")
    submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("AI is thinking..."):
            try:
                ai_response = run_chat(user_input)
            except Exception as e:
                ai_response = "⚠️ Sorry, something went wrong. Please try again later."
                st.error(f"Error: {e}")

        st.session_state.messages.append({"role": "ai", "content": ai_response})

# Update the placeholder with the current chat messages
with chat_placeholder:
    st.markdown("### 💬 Chat with the AI Therapist")
    for msg in st.session_state.messages:
        role = "🧑 You" if msg["role"] == "user" else "🤖 AI"
        st.markdown(f"**{role}:** {msg['content']}")
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="ChatGPT-Style IT Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Smart IT Support Chatbot")
st.caption("ChatGPT-style conversation for IT support questions.")

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_model()

SYSTEM_PROMPT = """
You are a helpful IT support assistant for a university.
You help with:
- password reset
- email issues
- printer issues
- LMS access
- WiFi, internet, VPN, and connectivity

Rules:
- Reply clearly and naturally.
- Be concise but helpful.
- Give practical steps.
- If the issue is unclear, ask one short clarifying question.
- If the issue seems account-specific or severe, advise contacting IT support.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I’m your IT Support Chatbot. How can I help you today?"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def build_prompt(messages):
    history = ""
    for msg in messages[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return f"""
{SYSTEM_PROMPT}

Conversation:
{history}

Assistant:
"""

user_input = st.chat_input("Type your IT question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    prompt = build_prompt(st.session_state.messages)

    try:
        result = generator(
            prompt,
            max_length=200,
            do_sample=True,
            temperature=0.7
        )

        reply = result[0]["generated_text"].strip()

        if not reply:
            reply = "Sorry, I could not generate a response. Please try again."

    except Exception:
        reply = "Sorry, something went wrong while generating the response."

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.write(reply)

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Smart IT Support Chatbot", page_icon="🤖", layout="centered")

# -----------------------
# NLTK downloads
# -----------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------
# Dataset
# -----------------------
dataset = [
    ("I forgot my password", "Password"),
    ("Reset my password", "Password"),
    ("Password not working", "Password"),
    ("Forgot login password", "Password"),
    ("My password is not accepted", "Password"),

    ("The printer is offline", "Printer"),
    ("Printer not working", "Printer"),
    ("Cannot print document", "Printer"),
    ("Printer is not printing", "Printer"),
    ("Cannot connect to office printer", "Printer"),

    ("Outlook is not receiving emails", "Email"),
    ("Email not sending", "Email"),
    ("My email is not working", "Email"),
    ("Cannot receive messages", "Email"),
    ("Mailbox full", "Email"),

    ("I cannot access my course", "LMS"),
    ("LMS not loading", "LMS"),
    ("Course page not opening", "LMS"),
    ("LMS login not working", "LMS"),
    ("Cannot submit assignment", "LMS"),

    ("WiFi is not working", "Connectivity"),
    ("Internet connection issue", "Connectivity"),
    ("Cannot connect to WiFi", "Connectivity"),
    ("Internet is not connecting", "Connectivity"),
    ("WiFi keeps disconnecting", "Connectivity"),
]

responses = {
    "Password": "It looks like a password issue. Please reset your password using the university portal. If the problem continues, contact IT support.",
    "Printer": "It looks like a printer issue. Please check the printer power, paper, and network connection. You may also need to reinstall the printer driver.",
    "Email": "It looks like an email issue. Please check your Outlook settings and credentials. If the problem continues, restart Outlook or contact IT support.",
    "LMS": "It looks like an LMS issue. Please clear your browser cache and try logging in again. If the issue continues, contact academic IT support.",
    "Connectivity": "It looks like a connectivity issue. Please check your WiFi, VPN, or internet connection and try reconnecting."
}

candidate_labels = ["Password", "Printer", "Email", "LMS", "Connectivity"]

# -----------------------
# Preprocessing
# -----------------------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

all_words = sorted({word for text, label in dataset for word in preprocess(text)})

def extract_features(tokens):
    token_set = set(tokens)
    return {word: (word in token_set) for word in all_words}

# -----------------------
# Local model
# -----------------------
featuresets = [(extract_features(preprocess(text)), label) for text, label in dataset]
classifier = nltk.NaiveBayesClassifier.train(featuresets)

# -----------------------
# Hugging Face model
# -----------------------
@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

zero_shot = load_zero_shot()

# -----------------------
# UI style
# -----------------------
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 850px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Smart IT Support Chatbot")
st.caption("Ask about password, printer, email, LMS, or connectivity issues.")

# -----------------------
# Session state
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I’m your IT Support Chatbot. How can I help you today?"
        }
    ]

# -----------------------
# Show conversation
# -----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

# -----------------------
# Chat input
# -----------------------
user_input = st.chat_input("Type your IT question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    tokens = preprocess(user_input)
    features = extract_features(tokens)
    predicted = classifier.classify(features)
    confidence = classifier.prob_classify(features).prob(predicted)

    final_label = predicted
    final_confidence = confidence
    source = "Local model"

    # Use Hugging Face if local confidence is weak
    if confidence < 0.60:
        hf_result = zero_shot(user_input, candidate_labels)
        final_label = hf_result["labels"][0]
        final_confidence = hf_result["scores"][0]
        source = "Hugging Face zero-shot model"

    # Final response
    if final_confidence < 0.40:
        reply = (
            "Sorry, I am not fully sure about your issue. "
            "Please rephrase your question or contact IT support."
        )
    else:
        reply = responses.get(final_label, "Please contact IT support.")

    meta = f"Detected issue: {final_label} | Confidence: {final_confidence:.2f} | Source: {source}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "meta": meta
    })

    with st.chat_message("assistant"):
        st.write(reply)
        st.caption(meta)

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# NLTK downloads
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Dataset
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
    "Password": "Please reset your password using the university portal or contact IT support.",
    "Printer": "Please check the printer connection, paper, and driver settings.",
    "Email": "Please check your Outlook settings or email credentials.",
    "LMS": "Please try clearing your browser cache and logging in again.",
    "Connectivity": "Please check your WiFi or network connection and try again."
}

# Preprocess
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

# Train Naive Bayes
all_words = sorted({word for text, label in dataset for word in preprocess(text)})

def extract_features(tokens):
    token_set = set(tokens)
    return {word: (word in token_set) for word in all_words}

featuresets = [(extract_features(preprocess(text)), label) for text, label in dataset]
classifier = nltk.NaiveBayesClassifier.train(featuresets)

# Hugging Face zero-shot model
@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

zero_shot = load_zero_shot()

candidate_labels = ["Password", "Printer", "Email", "LMS", "Connectivity"]

st.title("Smart IT Support Chatbot")
st.write("Ask an IT-related question below.")

user_input = st.text_input("Your question:")

if user_input:
    # Step 1: local classifier
    tokens = preprocess(user_input)
    features = extract_features(tokens)
    predicted = classifier.classify(features)
    confidence = classifier.prob_classify(features).prob(predicted)

    st.write(f"**Local Prediction:** {predicted}")
    st.write(f"**Local Confidence:** {confidence:.2f}")

    # Step 2: if low confidence, use Hugging Face
    if confidence < 0.60:
        result = zero_shot(user_input, candidate_labels)
        hf_label = result["labels"][0]
        hf_score = result["scores"][0]

        st.write(f"**HF Prediction:** {hf_label}")
        st.write(f"**HF Confidence:** {hf_score:.2f}")

        if hf_score < 0.40:
            st.warning("Sorry, I am not fully sure about your issue. Please rephrase your question or contact IT support.")
        else:
            st.success(responses.get(hf_label, "Please contact IT support."))
    else:
        st.success(responses.get(predicted, "Please contact IT support."))

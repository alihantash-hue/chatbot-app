import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

dataset = [
    ("I forgot my password", "Password"),
    ("Reset my password", "Password"),
    ("Password not working", "Password"),
    ("The printer is offline", "Printer"),
    ("Printer not working", "Printer"),
    ("Cannot print document", "Printer"),
    ("Outlook is not receiving emails", "Email"),
    ("Email not sending", "Email"),
    ("My email is not working", "Email"),
    ("I cannot access my course", "LMS"),
    ("LMS not loading", "LMS"),
    ("Course page not opening", "LMS"),
    ("WiFi is not working", "Connectivity"),
    ("Internet connection issue", "Connectivity"),
    ("Cannot connect to WiFi", "Connectivity"),
]

responses = {
    "Password": "Please reset your password using the university portal or contact IT support.",
    "Printer": "Please check the printer connection, paper, and driver settings.",
    "Email": "Please check your Outlook settings or email credentials.",
    "LMS": "Please try clearing your browser cache and logging in again.",
    "Connectivity": "Please check your WiFi or network connection and try again."
}

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

featuresets = [(extract_features(preprocess(text)), label) for text, label in dataset]
classifier = nltk.NaiveBayesClassifier.train(featuresets)

st.title("IT Support Chatbot")
st.write("Ask an IT-related question below.")

user_input = st.text_input("Your question:")

if user_input:
    tokens = preprocess(user_input)
    features = extract_features(tokens)
    predicted = classifier.classify(features)
    confidence = classifier.prob_classify(features).prob(predicted)

    st.write(f"**Predicted Intent:** {predicted}")
    st.write(f"**Confidence:** {confidence:.2f}")

    if confidence < 0.5:
        st.warning("Sorry, I am not fully sure about your issue. Please contact IT support.")
    else:
        st.success(responses.get(predicted, "Please contact IT support."))

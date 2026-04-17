import streamlit as st
import re

st.set_page_config(page_title="Humanizer App", page_icon="🧠")

st.title("🧠 Humanizer App")
st.write("Improve clarity and make text sound more natural without changing meaning.")

text = st.text_area("Enter your text here:")

def humanize(text):
    text = re.sub(r"\s+", " ", text)

    replacements = {
        "In conclusion": "To sum up",
        "It is important to note that": "Importantly,",
        "Due to the fact that": "Because",
        "In order to": "To",
        "A large number of": "Many",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

if st.button("Humanize"):
    if text.strip():
        result = humanize(text)
        st.subheader("Improved Version")
        st.write(result)
    else:
        st.warning("Please enter some text.")

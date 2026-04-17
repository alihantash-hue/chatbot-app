import streamlit as st
import re

# Page settings
st.set_page_config(page_title="Humanizer App", page_icon="🧠")

# Title
st.title("🧠 Humanizer App")
st.write("Improve clarity and make text sound more natural without changing meaning.")

# Tone selector
tone = st.selectbox(
    "Select tone:",
    ["Natural", "Professional", "Academic", "Simple"]
)

# Input
text = st.text_area("Enter your text here:")

# Humanizer function
def humanize(text, tone):
    text = re.sub(r"\s+", " ", text)

    if tone == "Natural":
        replacements = {
            "In conclusion": "So",
            "It is important to note that": "Importantly,",
            "Due to the fact that": "Because",
            "In order to": "To",
        }

    elif tone == "Professional":
        replacements = {
            "In conclusion": "In summary",
            "Due to the fact that": "Because",
            "A large number of": "Numerous",
            "In order to": "To",
        }

    elif tone == "Academic":
        replacements = {
            "Because": "Due to the fact that",
            "Many": "A significant number of",
            "So": "Therefore",
        }

    else:  # Simple
        replacements = {
            "In conclusion": "So",
            "Due to the fact that": "Because",
            "A large number of": "Many",
            "In order to": "To",
        }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# Button
if st.button("Humanize"):
    if text.strip():
        result = humanize(text, tone)

        st.subheader("Improved Version:")
        st.write(result)

        # Extra feature
        st.caption(f"Word count: {len(result.split())}")
    else:
        st.warning("Please enter some text.")

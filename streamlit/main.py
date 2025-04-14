import streamlit as st
import nltk
import pickle
import time
import numpy as np
from keras.models import load_model
import json
import random

nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("popular")
from nltk.stem import WordNetLemmatizer

# Load your data and models
lemmatizer = WordNetLemmatizer()
model = load_model("./model.h5")
intents = json.loads(open("./gita_intents.json", encoding="utf-8").read())
words = pickle.load(open("./texts.pkl", "rb"))
classes = pickle.load(open("./labels.pkl", "rb"))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create a short form for the word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if the current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by the strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    global result
    try:
        tag = ints[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    except:
        result = "I cannot understand this statement. Perhaps rephrase it or type it differently?"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res

st.set_page_config(
    page_title="KãnhãSays",
    page_icon="🦚",
    layout="centered"  # or "wide"
)

def main():
    
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image("./static/krishna2.png", width=220, caption="Shree Krishna")

    st.markdown("<h1 style='text-align: center;'>KãnhãSays</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>The Divine Wisdom Of Shree Krishna - Your Key To Happiness🦚</h5>", unsafe_allow_html=True)

    # Define example questions
    example_questions = [
        "What is the meaning of life?",
        "How to deal with stress?",
        "What does the Gita say about karma?",
        "What is true happiness?",
    ]

    # State to manage question
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    # Show example questions as buttons
    st.markdown("<p style='text-align: center;'>Try asking:</p>", unsafe_allow_html=True)
    cols = st.columns(len(example_questions))
    for i, q in enumerate(example_questions):
        if cols[i].button(q):
            st.session_state.current_question = q  # Update session state

    # Question input area (pre-filled if user clicked a suggestion)
    question = st.text_area("Ask me something:", value=st.session_state.current_question, key="question_input")

    # Ask button logic
    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            time.sleep(3)  # Simulating processing time
            answer = chatbot_response(question)  # Replace with your model call

        # Fade-in answer display
        st.markdown(f"""
            <div style='text-align: center; animation: fadeIn 1s forwards; margin-top: 10px; font-size: 18px;'>{answer}</div>
            <style>
                @keyframes fadeIn {{
                    from {{ opacity: 0; }}
                    to {{ opacity: 1; }}
                }}
            </style>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

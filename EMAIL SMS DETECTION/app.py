import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()

# Load pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    # Join the processed tokens into a single string
    processed_text = ' '.join(y)

    return processed_text  # Return a single string



st.title("SMS SPAM CLASSIFIER")

input_sms = st.text_area("Enter the message")
if st.button('predict'):
    # Preprocess
    transform_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transform_sms])
    # Predict
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

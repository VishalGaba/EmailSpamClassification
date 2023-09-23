import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps=PorterStemmer()


def tranform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # for removing stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    #     Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

cv=pickle.load(open('vectorizer.pkl', 'rb'))
model=pickle.load(open('model.pkl', 'rb'))


st.title('Email/SMS spam classifier')

input_sms = st.text_input('enter the message')

if st.button('Predict'):


    tran_sms=tranform_text(input_sms)

    vector_input=cv.transform([tran_sms])

    result=model.predict(vector_input)[0]

    if result == 1:

        st.header("spam")
    else:
        st.header('Not spam')


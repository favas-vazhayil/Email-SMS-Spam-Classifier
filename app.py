import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps =PorterStemmer()
def transform_text(text):
    text=text.lower()   #Converting text into lower case
    text=nltk.word_tokenize(text) #Tokenization
    
    y=[]
    for i in text: #removing other characters from alphanumeric
        if i.isalnum():
            y.append(i)
    
    text=y[:] #Cloning y into text so that we dont have to define 
    y.clear() #a new empty list
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter the message ")

#Following are the steps 
if st.button('Predict'):
    #1.Preprocess
    transformed_sms = transform_text(input_sms)
    #2.Vectorization
    vector_input=tfidf.transform([transformed_sms])
    #3.Prediction
    result = model.predict(vector_input)[0]
    #4.Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")


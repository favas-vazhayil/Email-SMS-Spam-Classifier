import streamlit as st
import pickle
import string
import nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()

page_bg_img = '''
<style>
body {
background-image: url("https://media.istockphoto.com/id/1388948315/photo/abstract-colorful-background-brush-strokes-red-orange-yellow-purple-and-blue-colors-artistic.jpg?s=1024x1024&w=is&k=20&c=gj-8tkXjf-9_6gdXEEGAoj-X9ltZ0q4T9v8m7skdY-c=");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

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


import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#step-1 load the model and the word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
# Load the trained model
model = load_model('imdb_rnn_model.h5')
#step-2 helper function to decode the review back to text
def decode_review(encoded_review):
    return

#function to preprocess the review
def preprocess_review(review):
    words = review.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#step-3 prediction function
def predict_review(review):
    #preprocess the review
    processed_review=preprocess_review(review)
    #make prediction using the model
    prediction=model.predict(processed_review)
    sentement=  'positive' if prediction[0][0]>0.5 else 'negative'
    #return the predicted sentiment
    return sentement,prediction[0][0]

#streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")
# user input
user_input = st.text_area("Movie Review")

if st.button("Predict Sentiment"):
    preprocess_input = preprocess_review(user_input)
    #predict the sentiment
    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    st.write(f"Predicted Sentiment: {sentiment} (Confidence: {prediction[0][0]:.2f})")
    st.write(f'sentiment: {sentiment}, confidence: {prediction[0][0]:.2f}')
else:
    st.write("Please enter a movie review and click the 'Predict Sentiment' button.")
    

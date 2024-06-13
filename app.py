import streamlit as st
st.title("Sentiment Analysis of Reviews")
from model_predict import predict_sentiment
import pandas as pd
# File uploader for Excel sheet
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)

    # Check if the DataFrame contains a column named 'Review'
    if 'Review' in df.columns:
        # Apply the model_predict function to each review
        df['Sentiment'] = df['Review'].apply(predict_sentiment)

        # Display the DataFrame with sentiments
        st.write(df)
    else:
        st.error("The uploaded file does not contain a 'Review' column.")
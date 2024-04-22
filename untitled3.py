# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JKlDUKBSOVSD7v24HgXrQTAxgp6ClErG
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# Load the data (assuming you have already cleaned and preprocessed the data)
positive_df = pd.read_csv("positive_reviews.csv")
negative_df = pd.read_csv("negative_reviews.csv")

# Define function to perform topic modeling
def perform_topic_modeling(text_list):
    tv = TfidfVectorizer(min_df=2, max_df=0.40, ngram_range=(2, 2))
    dtm = tv.fit_transform(text_list)
    lda = LatentDirichletAllocation(n_components=6, random_state=42)
    lda.fit(dtm)
    return lda, tv

# Perform topic modeling on positive and negative reviews
lda_positive, tv_positive = perform_topic_modeling(positive_df['Clean_All'].tolist())
lda_negative, tv_negative = perform_topic_modeling(negative_df['Clean_All'].tolist())

# Define function to get top 5 reviews for selected topic
def get_top_reviews(lda_model, text_list, selected_topic):
    dtm = lda_model.transform(tv.transform(text_list))
    top_topic_reviews = dtm[:, selected_topic].argsort()[-5:][::-1]
    return [text_list[i] for i in top_topic_reviews]

# Streamlit app
def main():
    st.title("Ryanair Customer Review Analyzer")
    option = st.sidebar.selectbox("Select Review Type", ("Positive", "Negative"))

    if option == "Positive":
        st.subheader("Positive Reviews")
        topic_options = range(6)
        selected_topic = st.sidebar.selectbox("Select Topic", topic_options)
        top_reviews = get_top_reviews(lda_positive, positive_df['Clean_All'].tolist(), selected_topic)
        for i, review in enumerate(top_reviews):
            st.write(f"**Review {i+1}:**")
            st.write(review)
            st.markdown("---")

    elif option == "Negative":
        st.subheader("Negative Reviews")
        topic_options = range(6)
        selected_topic = st.sidebar.selectbox("Select Topic", topic_options)
        top_reviews = get_top_reviews(lda_negative, negative_df['Clean_All'].tolist(), selected_topic)
        for i, review in enumerate(top_reviews):
            st.write(f"**Review {i+1}:**")
            st.write(review)
            st.markdown("---")

if __name__ == "__main__":
    main()
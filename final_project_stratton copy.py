import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the preprocessed data
ryan_air_df = pd.read_csv("/content/ryanair_reviews.csv")

# Preprocess the data
ryan_air_df['All_Text'] = ryan_air_df['Comment title'] + ' ' + ryan_air_df['Comment']
ryan_air_df['Clean_All'] = ryan_air_df['All_Text'].apply(clean_text)

# Split the dataset into positive and negative dataframes
positive_df = ryan_air_df[ryan_air_df['Overall Rating'] >= 5]
negative_df = ryan_air_df[ryan_air_df['Overall Rating'] < 5]

# Convert the text data into lists
text_list_pos = positive_df['Clean_All'].tolist()
text_list_neg = negative_df['Clean_All'].tolist()

# Initialize TF-IDF vectorizers
tv_pos = TfidfVectorizer(min_df=2, max_df=0.40, ngram_range=(2, 2))
tv_neg = TfidfVectorizer(min_df=2, max_df=0.40, ngram_range=(2, 2))

# Fit the vectorizers to positive and negative reviews
tfidf_matrix_pos = tv_pos.fit_transform(text_list_pos)
tfidf_matrix_neg = tv_neg.fit_transform(text_list_neg)

# Define functions for recommendation
def get_recommendations(query, tfidf_matrix, review_list):
    query_vector = tv_pos.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    similar_review_idxs = cosine_similarities[0].argsort()[-5:][::-1]
    similar_reviews = review_list[similar_review_idxs]
    return similar_reviews

# Streamlit app
def main():
    st.title("Ryanair Review Recommendation System")
    option = st.sidebar.selectbox("Select reviews type:", ("Positive Reviews", "Negative Reviews"))

    if option == "Positive Reviews":
        st.header("Positive Reviews")
        query = st.text_input("Enter your query:")
        if st.button("Get Recommendations"):
            if query:
                recommendations = get_recommendations(query, tfidf_matrix_pos, np.array(text_list_pos))
                st.subheader("Top 5 Customer Reviews:")
                for i, review in enumerate(recommendations, start=1):
                    st.write(f"{i}. {review}")

    elif option == "Negative Reviews":
        st.header("Negative Reviews")
        query = st.text_input("Enter your query:")
        if st.button("Get Recommendations"):
            if query:
                recommendations = get_recommendations(query, tfidf_matrix_neg, np.array(text_list_neg))
                st.subheader("Top 5 Customer Reviews:")
                for i, review in enumerate(recommendations, start=1):
                    st.write(f"{i}. {review}")

if __name__ == "__main__":
    main()


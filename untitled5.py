# import streamlit as st
# import pandas as pd
# import numpy as np
# import nltk
# import re
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import LatentDirichletAllocation

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# Load stopwords once to improve efficiency
stop_words = set(stopwords.words('english'))

positive_topics = [
    "Positive Flight Experience and Service",
    "Friendly Cabin Crew and Value for Money",
    "Consistent Positive Experiences",
    "Good Value and Punctuality",
    "Efficient Boarding and Comfort",
    "Positive Overall Experience and Recommendations"
]

negative_topics = [
    "Poor Customer Service and Flight Delays",
    "Online Check-in and Luggage Problems",
    "General Dissatisfaction with Ryanair Experience",
    "Luggage and Boarding Pass Hassles",
    "Payment Complaints and Boarding Issues",
    "Cabin Crew Service and Extra Charges"
]

def clean_text(text, stop_words):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(token) for token in tokens]
    cleaned_text = ' '.join(lemmatized_text)
    return cleaned_text

def display_positive_topics_and_reviews(lda, tfidf_matrix_pos, df_doc_topic):
    st.title('Positive Reviews Topics and Recommendations')
    st.write("Select a topic:")
    # Use custom labels for positive topics
    selected_topic = st.selectbox("Topics", positive_topics, key='positive_topic')
    top_reviews = display_top_reviews(selected_topic, positive_df, tfidf_matrix_pos, lda, df_doc_topic)
    st.write("Top 5 Customer Reviews:")
    for review in top_reviews:
        st.write(review)

def display_negative_topics_and_reviews(lda, tfidf_matrix_neg, df_doc_topic):
    st.title('Negative Reviews Topics and Recommendations')
    st.write("Select a topic:")
    # Use custom labels for negative topics
    selected_topic = st.selectbox("Topics", negative_topics, key='negative_topic')
    top_reviews = display_top_reviews(selected_topic, negative_df, tfidf_matrix_neg, lda, df_doc_topic)
    st.write("Top 5 Customer Reviews:")
    for review in top_reviews:
        st.write(review)

def query_review_recommender(search_query, reviews, tfidf_vectorizer, tfidf_matrix):
    query_vector = tfidf_vectorizer.transform([search_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    # Get indices sorted by similarity score
    similar_review_idxs = cosine_similarities[0].argsort()[-5:][::-1]
    # Ensure indices do not exceed the length of reviews
    similar_review_idxs = [idx for idx in similar_review_idxs if idx < len(reviews)]
    similar_review = reviews[similar_review_idxs]
    return similar_review


def display_top_reviews(selected_topic, df, tfidf_matrix, lda, df_doc_topic):
    # Get the index of the selected topic from its description
    if selected_topic in positive_topics:
        topic_index = positive_topics.index(selected_topic)
    else:
        topic_index = negative_topics.index(selected_topic)
    
    topic_reviews = df.iloc[df_doc_topic[df_doc_topic[f'Topic {topic_index}'] == df_doc_topic[f'Topic {topic_index}'].max()].index]['Comment']
    similar_reviews = query_review_recommender(topic_reviews.iloc[0], df['Comment'].values, tf, tfidf_matrix)
    return similar_reviews

# Load data
ryan_air_df = pd.read_csv("/Users/strattonbutterfield/Documents/ryanair_reviews.csv")
ryan_air_df['All_Text'] = ryan_air_df['Comment title'] + ' ' + ryan_air_df['Comment']
ryan_air_df['Clean_All'] = ryan_air_df['All_Text'].apply(lambda x: clean_text(x, stop_words))
positive_df = ryan_air_df[ryan_air_df['Overall Rating'] >= 5]
negative_df = ryan_air_df[ryan_air_df['Overall Rating'] < 5]

text_list_pos = positive_df['Clean_All'].tolist()
text_list_neg = negative_df['Clean_All'].tolist()

# TF-IDF vectorization
tf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.80)
tfidf_matrix = tf.fit_transform(ryan_air_df['Clean_All'])

# Latent Dirichlet Allocation
lda = LatentDirichletAllocation(n_components=6, random_state=42)
lda.fit(tfidf_matrix)

# Document-topic matrix
doc_topic_matrix = lda.transform(tfidf_matrix)
df_doc_topic = pd.DataFrame(doc_topic_matrix, columns=[f'Topic {i}' for i in range(lda.n_components)])

# Streamlit app layout
st.title('Ryanair Customer Reviews')
function_selection = st.radio("Choose Functionality", ("Positive Reviews", "Negative Reviews"))

if function_selection == "Positive Reviews":
    display_positive_topics_and_reviews(lda, tfidf_matrix, df_doc_topic)
elif function_selection == "Negative Reviews":
    display_negative_topics_and_reviews(lda, tfidf_matrix, df_doc_topic)

def ask_question_and_recommend_reviews():
    st.title('Review Recommender. Find specific Customer Reviews Based on Specific Keywords.')
    question = st.text_input('Input your keword here:', '')
    if st.button('Ask'):
        if question:
            similar_reviews = query_review_recommender(question, ryan_air_df['Comment'].values, tf, tfidf_matrix)
            st.write("Top 5 Most Similar Customer Reviews:")
            for review in similar_reviews:
                st.write(review)

ask_question_and_recommend_reviews()


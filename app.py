import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ Dataset
faq_data = {
    "Question": [
        "What is AI?",
        "What is Machine Learning?",
        "What is Python?",
        "What is Streamlit?",
        "What is NLP?"
    ],
    "Answer": [
        "AI stands for Artificial Intelligence.",
        "Machine Learning is a subset of AI.",
        "Python is a programming language.",
        "Streamlit is a Python framework for web apps.",
        "NLP stands for Natural Language Processing."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(faq_data)

# Streamlit UI
st.title("🤖 FAQ Chatbot")

user_question = st.text_input("Ask a question:")

if user_question:
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df["Question"].tolist() + [user_question])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    best_match_index = similarity.argmax()

    response = df.iloc[best_match_index]["Answer"]

    st.success(response)
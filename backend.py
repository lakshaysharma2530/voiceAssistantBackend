import nltk
nltk.download('stopwords')
nltk.download('punkt')

from fastapi import FastAPI
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CSV file containing Q&A
file_path = Path(__file__).resolve().parent /'Training.csv'
data = pd.read_csv(file_path)

# Preprocess the text data
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercasing
    tokens = [stemmer.stem(t) for t in tokens if t.isalnum()]  # Stemming
    tokens = [t for t in tokens if t not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

data['Processed_Question'] = data['Question'].apply(preprocess_text)

# Vectorize the text data
vectorizer = TfidfVectorizer()
vectorized_data = vectorizer.fit_transform(data['Processed_Question'])

# Function to find the most relevant answer
def get_answer(query):
    processed_query = preprocess_text(query)
    query_vectorized = vectorizer.transform([processed_query])

    # Calculate cosine similarity between the query and the stored questions
    similarities = cosine_similarity(query_vectorized, vectorized_data)
    most_similar_index = similarities.argmax()
    print(f"Cosine Similarity: {similarities[0, most_similar_index]}")
    return data['Answer'][most_similar_index]

@app.get("/")
def read_root():
    return {"message": "Hello, I'm SHARDA Assistant. How may I help you?"}

@app.get("/ask")
def ask_question(query: str):
    # Example usage: /ask?query=How do I reset my password?
    print(query)
    answer = get_answer(query)
    print(answer)
    return {"question": query, "answer": answer}

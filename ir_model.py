import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess the text by removing non-alphanumeric characters, digits, converting to lowercase, etc."""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

def load_texts_from_folder(path):
    """Load all text files from a folder."""
    documents = []
    file_names = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(preprocess_text(file.read()))
                file_names.append(filename)
    return file_names, documents

def create_vector_space(documents):
    """Create the TF-IDF vector space representation of documents."""
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

def rank_documents(query, vectorizer, tfidf_matrix, titles):
    """Rank documents based on cosine similarity with the query."""
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = cosine_similarities.argsort()[::-1]
    top_indices = ranked_indices[:5]
    return [(titles[i], cosine_similarities[i]) for i in top_indices]

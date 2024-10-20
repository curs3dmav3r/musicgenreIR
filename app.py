from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ir_model import load_texts_from_folder, create_vector_space, rank_documents, preprocess_text

# Initialize FastAPI app
app = FastAPI()

# Load documents and initialize the vectorizer
docs_folder_path = "/Users/manish/Desktop/informationretrieval/week2assignment/musicgenreIRapp/dataset/"  # Make sure this is the path where your documents are stored
titles, documents = load_texts_from_folder(docs_folder_path)
vectorizer, tfidf_matrix = create_vector_space(documents)

# Define request model
class Query(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Information Retrieval API!"}

@app.post("/search")
def search(query: Query):
    # Preprocess the query
    processed_query = preprocess_text(query.query)
    
    # Rank the documents based on the query
    try:
        results = rank_documents(processed_query, vectorizer, tfidf_matrix, titles)
        return {"query": query.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from huggingface_hub import InferenceClient


# Load FAISS index and metadata
index = faiss.read_index('notebooks/vector_store/complaints_faiss.index')
metadata = pd.read_csv('notebooks/vector_store/metadata.csv')

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_similar_chunks(question, k=5):
    # Embed the question
    question_embedding = model.encode([question])
    # Search FAISS index
    D, I = index.search(np.array(question_embedding).astype('float32'), k)
    # Get the top-k chunks and their metadata
    results = metadata.iloc[I[0]]
    return results

def generate_answer(question, retrieved_chunks):
    # Combine the top chunks into context
    context = "\n".join(retrieved_chunks['chunk'].tolist())
    llm = InferenceClient(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                token=os.getenv("hugging-face-token")
            )
        
    prompt = (
        f"You are a financial analyst assistant for CrediTrust. "
        f"Your task is to answer questions about customer complaints. "
        f"Use the following retrieved complaint excerpts to formulate your answer. "
        f"If the context doesn't contain the answer, state that you don't have enough information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                stop=["\n\n"]
            )
    # Use a small model for demo; replace with a better one if needed
    #qa_pipeline = pipeline("text-generation", model="google/flan-t5-small", max_new_tokens=128)
    #answer = qa_pipeline(prompt)[0]['generated_text']
    answer = response.choices[0].message.content.strip()
    return answer

# Example usage:
if __name__ == "__main__":
    question = "Why are people unhappy with Buy Now, Pay Later?"
    top_chunks = retrieve_similar_chunks(question, k=5)
    print("Retrieved Chunks:")
    print(top_chunks[['product', 'chunk']])
    print("\nGenerated Answer:")
    print(generate_answer(question, top_chunks))
   

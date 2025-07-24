# CrediTrust Complaint Insights Chatbot

This project provides a Streamlit-based chatbot for exploring and analyzing customer complaints in the financial services sector. It uses Retrieval-Augmented Generation (RAG) to answer questions based on complaint data.

## Folder Structure

- `app.py`  
  Main Streamlit app for the chatbot interface.

- `src/`  
  - `preprocess.py`: Data preprocessing scripts.  
  - `rag_pipeline.py`: RAG pipeline for retrieval and answer generation.

- `data/`  
  Contains CSV files for complaints and processed data.

- `notebooks/`  
  Jupyter notebooks for EDA and preprocessing.

- `vector_store/`  
  FAISS index and metadata for fast similarity search.

- `chatbot/`  
  Python virtual environment (do not edit).

## Getting Started

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the app**  
   ```
   streamlit run app.py
   ```

3. **Data Preparation**  
   - Use notebooks in `notebooks/` for EDA and chunking.
   - Ensure processed data is in `data/` and FAISS index in `vector_store/`.

## Usage

- Enter a question about customer complaints in the input box.
- Click "Submit" to get an answer and see relevant complaint chunks as sources.
- Click "Clear" to reset the input.

## Notes

- The virtual environment is in `chatbot/` (excluded from version control).
- Data files and vector store are also excluded via `.gitignore`.

## License
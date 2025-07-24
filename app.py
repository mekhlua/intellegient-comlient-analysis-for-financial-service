import streamlit as st
from src.rag_pipeline import retrieve_similar_chunks, generate_answer

st.title("CrediTrust Complaint Insights Chatbot")

# Always show the input box
question = st.text_input("Ask a question about customer complaints:")

# Show buttons in a row
col1, col2 = st.columns([1, 1])
submit = col1.button("Submit")
clear = col2.button("Clear")

if clear:
    st.experimental_rerun()

if submit and question:
    with st.spinner("Retrieving and generating answer..."):
        top_chunks = retrieve_similar_chunks(question, k=5)
        answer = generate_answer(question, top_chunks)
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("### Sources")
        for i, row in top_chunks.iterrows():
            st.write(f"**Product:** {row['product']}")
            st.write(f"**Company:** {row.get('company', '')}")
            st.write(f"**Date:** {row.get('date', '')}")
            st.write(row['chunk'])
            st.markdown("---")
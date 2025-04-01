import streamlit as st
from langchain_core.messages import HumanMessage

# Load the chain from the existing RAG pipeline
from main import chain_multimodal_rag

# Streamlit UI setup
st.title("AI-Powered Q&A System")
st.write("Ask any question, and the model will provide an answer based on the retrieved documents.")

# User input field
user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Retrieving information..."):
        response = chain_multimodal_rag.invoke(user_question)
    
    st.subheader("Answer:")
    st.write(response)

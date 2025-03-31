import streamlit as st
from PIL import Image
import base64
import os
import re
import io
import uuid
from dotenv import load_dotenv

# Import required modules from your existing code
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Load environment variables
load_dotenv()

# Set API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Functions from your existing code
def extract_pdf_elements(file_bytes, extract_images=False):
    """Extract elements from uploaded PDF file"""
    # Save temporary file
    temp_file = "temp_uploaded.pdf"
    with open(temp_file, "wb") as f:
        f.write(file_bytes)
    
    # Process PDF
    elements = partition_pdf(
        filename=temp_file,
        extract_images_in_pdf=extract_images,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path="./",
    )
    
    # Clean up temp file
    os.remove(temp_file)
    return elements

def categorize_elements(raw_pdf_elements):
    """Categorize elements by type"""
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

def generate_summaries(tables):
    """Summarize table elements only"""
    prompt_text = """You are an assistant tasked with summarizing tables for retrieval. \
    These summaries will be embedded and used to retrieve the raw table elements. \
    Give a concise summary of the table that is well optimized for retrieval. Table: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatGroq(model_name='llama3-70b-8192')
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    table_summaries = []
    if tables:
        with st.spinner("Generating table summaries..."):
            table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})

    return table_summaries

def encode_image(image_bytes):
    """Encode image bytes to base64"""
    return base64.b64encode(image_bytes).decode("utf-8")

def image_summarize(img_base64, prompt):
    """Generate summary for an image"""
    chat = ChatGroq(model_name='gemma2-9b-it')
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                ]
            )
        ]
    )
    return msg.content

def looks_like_base64(sb):
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    text_message = {
        "type": "text",
        "text": (
            "You are a physics expert tasked with providing clear explanations.\n"
            "Use the following information to provide a comprehensive but non-repetitive answer to the user question.\n"
            f"User question: {data_dict['question']}\n\n"
            "Reference information:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

def create_multi_vector_retriever(vectorstore, texts, table_summaries, tables, image_summaries, images):
    """Create a multi-vector retriever with texts, tables, and images"""
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    def add_documents(retriever, doc_summaries_or_texts, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        if isinstance(doc_summaries_or_texts[0], Document):
            docs = doc_summaries_or_texts
        else:
            docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) 
                   for i, s in enumerate(doc_summaries_or_texts)]
        retriever.vectorstore.add_documents(docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add raw texts directly, summaries for tables and images
    if texts:
        add_documents(retriever, texts, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever

def multi_modal_rag_chain(retriever):
    """Create a multimodal RAG chain"""
    model = ChatGroq(model_name='llama3-70b-8192')
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    return chain

def process_uploaded_images(uploaded_images):
    """Process uploaded images and generate summaries"""
    img_base64_list = []
    image_summaries = []
    
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    Give a concise summary of the image that is well optimized for retrieval."""
    
    with st.spinner("Processing images..."):
        for img_file in uploaded_images:
            img_bytes = img_file.getvalue()
            base64_image = encode_image(img_bytes)
            img_base64_list.append(base64_image)
            
            # Display processed image
            st.image(img_bytes, caption=f"Processed: {img_file.name}", width=300)
            
            # Generate summary
            summary = image_summarize(base64_image, prompt)
            image_summaries.append(summary)
            st.write(f"Summary: {summary}")
    
    return img_base64_list, image_summaries

def initialize_app():
    """Initialize the Streamlit app"""
    st.set_page_config(
        page_title="Physics RAG Assistant",
        page_icon="🔬",
        layout="wide"
    )
    
    # Initialize session state for storing the retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        st.session_state.rag_chain = None
        st.session_state.has_data = False
    
    # App title and description
    st.title("Physics RAG Assistant 🔬")
    st.markdown("""
    Upload physics documents and ask questions to get comprehensive answers. 
    This system uses Retrieval-Augmented Generation (RAG) with Langchain and Groq LLMs.
    """)

def main():
    # Initialize app
    initialize_app()
    
    # Sidebar for uploads and settings
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_pdfs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
        
        extract_images = st.checkbox("Extract images from PDFs", value=False)
        
        st.header("Upload Additional Images")
        uploaded_images = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        process_button = st.button("Process Documents")
        
        if st.session_state.has_data:
            st.success("✅ Documents processed and ready for questions!")
    
    # Main area for asking questions
    if not st.session_state.has_data:
        st.info("Please upload documents and click 'Process Documents' to get started.")
    
    # Process documents when button is clicked
    if process_button and (uploaded_pdfs or uploaded_images):
        texts_4k_token = []
        tables = []
        table_summaries = []
        img_base64_list = []
        image_summaries = []
        
        # Process PDFs
        if uploaded_pdfs:
            for pdf_file in uploaded_pdfs:
                st.write(f"Processing {pdf_file.name}...")
                
                # Extract elements from PDF
                raw_pdf_elements = extract_pdf_elements(pdf_file.getvalue(), extract_images)
                
                # Categorize elements
                pdf_texts, pdf_tables = categorize_elements(raw_pdf_elements)
                
                # Split texts into chunks
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=1000, chunk_overlap=100
                )
                pdf_texts_chunks = text_splitter.split_text(" ".join(pdf_texts))
                
                # Add to collection
                texts_4k_token.extend(pdf_texts_chunks)
                tables.extend(pdf_tables)
        
        # Generate table summaries
        if tables:
            table_summaries = generate_summaries(tables)
        
        # Process uploaded images
        if uploaded_images:
            img_base64_list, image_summaries = process_uploaded_images(uploaded_images)
        
        # Initialize embeddings and vector store
        with st.spinner("Setting up vector store..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            
            # Initialize FAISS with texts if available
            if texts_4k_token:
                vectorstore = FAISS.from_texts(
                    texts=texts_4k_token,
                    embedding=embeddings
                )
            else:
                vectorstore = FAISS.from_texts(
                    texts=["Initial empty document"],
                    embedding=embeddings
                )
        
        # Create retriever and RAG chain
        with st.spinner("Creating retriever and RAG chain..."):
            retriever = create_multi_vector_retriever(
                vectorstore,
                texts_4k_token,
                table_summaries,
                tables,
                image_summaries,
                img_base64_list
            )
            
            # Set retrieval limit
            retriever.search_kwargs["k"] = 5
            
            # Create RAG chain
            rag_chain = multi_modal_rag_chain(retriever)
            
            # Store in session state
            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.session_state.has_data = True
        
        st.success("Documents processed successfully! You can now ask questions.")
    
    # Question input area
    if st.session_state.has_data:
        st.header("Ask Questions About Your Physics Documents")
        user_question = st.text_input("Enter your question:")
        
        if user_question:
            with st.spinner("Generating answer..."):
                # Retrieve documents for context
                retrieved_docs = st.session_state.retriever.invoke(user_question)
                
                # Generate answer
                response = st.session_state.rag_chain.invoke(user_question)
                
                # Display answer
                st.header("Answer")
                st.write(response)
                
                # Optional: Show retrieved contexts in an expander
                with st.expander("View Retrieved Context"):
                    st.write(f"Retrieved {len(retrieved_docs)} documents for context.")
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Document {i+1}:**")
                        
                        # Check if it's an image
                        if looks_like_base64(doc.page_content) and is_image_data(doc.page_content):
                            st.image(f"data:image/jpeg;base64,{doc.page_content}", caption=f"Image {i+1}")
                        else:
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        st.markdown("---")

if __name__ == "__main__":
    main()
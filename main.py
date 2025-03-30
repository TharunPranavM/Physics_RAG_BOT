from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import base64
import os
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import io
import re
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Extract elements from PDF
def extract_pdf_elements(path, fname):
    return partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )

# Categorize elements by type
def categorize_elements(raw_pdf_elements):
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables

# File path 
fpath = r"D:\projects\phy_rag_bot" + "\\" 
fname = "chapters.pdf"

# Get elements
raw_pdf_elements = extract_pdf_elements(fpath, fname)

# Get text, tables
texts, tables = categorize_elements(raw_pdf_elements)

# Split texts into chunks without summarization
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)
texts_4k_token = text_splitter.split_text(" ".join(texts))

# Modified summary function - only for tables
def generate_summaries(tables):
    """
    Summarize table elements only
    tables: List of str
    """
    prompt_text = """You are an assistant tasked with summarizing tables for retrieval. \
    These summaries will be embedded and used to retrieve the raw table elements. \
    Give a concise summary of the table that is well optimized for retrieval. Table: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatGroq(model_name='llama3-70b-8192')
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    table_summaries = []
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})

    return table_summaries

# Get table summaries only
table_summaries = generate_summaries(tables)

# Image handling functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt):
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

def generate_img_summaries(path):
    img_base64_list = []
    image_summaries = []
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    Give a concise summary of the image that is well optimized for retrieval."""
    
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))
    
    return img_base64_list, image_summaries

# Get image summaries
img_base64_list, image_summaries = generate_img_summaries(fpath)

# Vector store setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize FAISS with an initial empty text or with texts if available
if texts_4k_token:
    vectorstore = FAISS.from_texts(
        texts=[""],  # Initial empty text to create the store
        embedding=embeddings
    )
else:
    vectorstore = FAISS.from_texts(
        texts=["Initial empty document"],
        embedding=embeddings
    )

# Modified retriever function
def create_multi_vector_retriever(
    vectorstore, texts, table_summaries, tables, image_summaries, images
):
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

# Create retriever with raw texts
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    texts_4k_token,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)

# Display and RAG chain functions
def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))

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

def multi_modal_rag_chain(retriever):
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


# Convert FAISS vectorstore to a retriever
retriever_multi_vector_img = vectorstore.as_retriever()

# Increase retrieval limit to get more relevant documents
retriever_multi_vector_img.search_kwargs["k"] = 5  # Retrieve more documents

# Create RAG chain
chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)


# Example question to test the model
example_question = "What is MAGNETIC FORCE ?"
response = chain_multimodal_rag.invoke(example_question)

# Debugging: Check how many documents are retrieved for a sample query
example_query = "Explain the concept of quantum entanglement."
retrieved_docs = retriever_multi_vector_img.invoke(example_query)
print(f"Retrieved {len(retrieved_docs)} documents.")
print("\n--- MODEL RESPONSE ---")
print(response)
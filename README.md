# 📘 Physics RAG Assistant

## 🔬 Overview
Physics RAG Assistant is an intelligent document processing and question-answering system designed to help users retrieve and analyze physics-related information. The application utilizes Retrieval-Augmented Generation (RAG) with Langchain and Groq LLMs to provide accurate and contextually relevant answers based on uploaded PDFs and images.

## 🚀 Features
- **📄 PDF Processing**: Extracts text, tables, and images from uploaded PDF documents.
- **🖼️ Image Summarization**: Analyzes and summarizes uploaded images for efficient retrieval.
- **📚 Multi-Vector Retrieval**: Leverages FAISS and Hugging Face embeddings to enhance document search.
- **🧠 AI-Powered Q&A**: Uses a multi-modal RAG pipeline to generate responses based on retrieved context.
- **📈 Table Summarization**: Extracts and summarizes tabular data for easier understanding.

## 🛠️ Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/physics-rag-assistant.git
   cd physics-rag-assistant
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the project root and add your API keys:
     ```env
     GROQ_API_KEY=your_groq_api_key
     HF_TOKEN=your_huggingface_api_key
     ```

## 🎯 Usage
1. Run the application:
   ```sh
   python app.py
   ```
2. Open the web application in your browser.
3. Upload PDFs and images from the sidebar.
4. Click "Process Documents" to analyze and store data.
5. Enter your physics-related question in the input field.
6. Get AI-generated answers based on retrieved documents and images.

## 🔄 Workflow
1. **Upload Documents & Images**: Users upload physics-related PDFs or images.
2. **Processing & Extraction**:
   - PDFs are parsed for text, tables, and images.
   - Images are analyzed and summarized.
3. **Vector Embedding & Storage**:
   - Extracted data is converted into vector embeddings.
   - FAISS is used to store and retrieve relevant documents.
4. **User Query Handling**:
   - Users enter queries related to physics concepts.
   - The system retrieves relevant documents using vector search.
5. **AI-Powered Response Generation**:
   - The retrieved data is passed through Langchain and Groq LLMs.
   - A final AI-generated response is provided to the user.

## 📂 Project Structure
```
📦 physics-rag-assistant
├── 📜 app.py              # Main application script
├── 📜 requirements.txt    # Dependencies
├── 📂 assets              # Static assets (if needed)
└── 📜 .env.example        # Environment variables template
```

## 🏗️ Technologies Used
- **Streamlit** for the user interface
- **Langchain** for AI-driven document processing
- **FAISS** for vector-based retrieval
- **Hugging Face** embeddings
- **Groq LLMs** for generating responses

---



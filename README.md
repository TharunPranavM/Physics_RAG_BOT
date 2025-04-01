# 📘 Physics RAG Assistant

## 🔬 Overview
Physics RAG Assistant is an advanced AI-powered document processing and question-answering system tailored for physics enthusiasts, students, and researchers. This intelligent assistant leverages Retrieval-Augmented Generation (RAG) with Langchain and Groq LLMs to retrieve and analyze information from uploaded PDFs and images, providing users with highly accurate, contextually relevant answers.

## 🚀 Features
### 🏆 Cutting-Edge Capabilities
- **📄 PDF Processing**: Extracts and processes text, tables, and images from uploaded physics-related PDFs, enabling precise information retrieval.
- **🖼️ Image Summarization**: Analyzes scientific images and diagrams, summarizing their content for enhanced understanding.
- **📚 Multi-Vector Retrieval**: Utilizes FAISS and Hugging Face embeddings to conduct intelligent and efficient document searches.
- **🧠 AI-Powered Q&A**: Implements a multi-modal RAG pipeline to answer complex physics questions with high accuracy.
- **📈 Table Summarization**: Extracts and summarizes key insights from tabular data, making numerical data easier to interpret.
- **⚡ Fast and Scalable**: Optimized for quick retrieval and large-scale document processing.

## 🛠️ Installation & Setup
### 📌 Prerequisites
Before installing, ensure you have the following:
- Python 3.8+
- pip (Python package manager)

### 🏗️ Installation Steps
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/your-repo/physics-rag-assistant.git
   cd physics-rag-assistant
   ```
2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables**:
   - Create a `.env` file in the project root and add the following keys:
     ```env
     GROQ_API_KEY=your_groq_api_key
     HF_TOKEN=your_huggingface_api_key
     ```

## 🎯 How to Use
1. **Run the Application**:
   ```sh
   streamlit run app.py
   ```
2. **Interact with the AI**:
   - Open the web app in your browser.
   - Type your physics-related question in the input field.
   - Receive AI-generated answers based on retrieved documents and images.

## 🔄 System Workflow
### 🚀 End-to-End Process
1. **📤 Upload Physics Documents & Images**
   - PDFs are parsed to extract text, tables, and images.
   - Images undergo AI-based summarization.
2. **🔍 Data Processing & Vector Storage**
   - Extracted content is converted into vector embeddings.
   - FAISS stores and retrieves relevant documents efficiently.
3. **💡 AI-Powered Answer Generation**
   - Users enter queries related to physics topics.
   - The system searches for relevant documents.
   - The retrieved content is processed by Langchain and Groq LLMs to generate precise responses.
4. **📜 Display Results**
   - The AI-generated answer is presented in an easy-to-read format.

## 📂 Project Structure
```
📦 physics-rag-assistant
├── 📜 app.py              # Streamlit application script (frontend & UI)
├── 📜 main.py             # Core multi-modal RAG pipeline
├── 📜 requirements.txt    # Dependency file for installation
└── 📜 .env                # Environment variable template
```

## 🏗️ Technologies & Tools Used
- **🖥️ Streamlit**: Interactive UI for the Q&A assistant.
- **🧠 Langchain**: Framework for AI-powered document processing.
- **📊 FAISS**: High-performance vector database for efficient document retrieval.
- **🔍 Hugging Face Embeddings**: State-of-the-art embeddings for better search accuracy.
- **📝 Groq LLMs**: AI model responsible for generating precise answers.

---


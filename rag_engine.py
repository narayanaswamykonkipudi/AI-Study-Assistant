import os
import pdfplumber
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"

def extract_text(pdf_file) -> str:
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def build_vector_store(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_rag_chain(vector_store: FAISS):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful study assistant.
Use ONLY the following context from the student's notes to answer.
If the answer is not in the context, say "I couldn't find that in your notes."

Context:
{context}

Question: {question}

Answer:"""
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def summarize_text(text: str) -> str:
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize the following study material into clear, 
concise bullet points organized by topic. Make it easy to revise quickly.

Material:
{text}

Summary:"""
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text[:4000]})
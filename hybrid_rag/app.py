from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from pinecone_text.sparse import BM25Encoder
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

import nltk
nltk.download('punkt_tab')

pinecone_api = os.environ['pinecone_api']
groq_api_key = os.environ['GROQ_API_KEY']

index_name = "hybrid-search-langchain-rag"

pc = Pinecone(api_key = pinecone_api)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension= 4096,
        metric='dotproduct',
        spec = ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)
print('111111111111111111')

embeddings = OllamaEmbeddings()
bm25encoder = BM25Encoder().default()

st.title("Hybrid-Search-RAG-Appliction")
text_inp = st.text_input("Enter Paragraph/sentences")
# text_inp = input()
texts = text_inp.split('.')
sentences = texts[:-1]
print("########",sentences)
bm25encoder.fit(sentences)
bm25encoder.dump("bm25.json")

bm25encoder = BM25Encoder().load("bm25.json")

print("2222##########")
retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25encoder, index=index)

print("333333#########")
retriever.add_texts(sentences)

print("44444444########")
question = st.text_input("Ask Question here...")

llm = ChatGroq(groq_api_key=groq_api_key,model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>

    Questions: {input}
    """
)

document_chain = create_stuff_documents_chain(llm,prompt)
retrieval_chain = create_retrieval_chain(retriever,document_chain)

response = retrieval_chain.invoke({"input":question})


if question:
    response = retrieval_chain.invoke({"input":question})
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
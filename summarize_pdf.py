import streamlit as st
from langchain.llms import HuggingFaceHub
import os
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


os.environ['HUGGINGFACEHUB_API_TOKEN'] ="some_key"

loader=PyPDFLoader("Atatürk's Address to the Turkish Youth.pdf")

docs=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

texts=text_splitter.split_documents(docs)

chain=load_summarize_chain(llm=HuggingFaceHub(repo_id="facebook/bart-large-cnn"),
                            chain_type='map_reduce')

print(chain.run(texts))



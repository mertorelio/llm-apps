import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = st.text_input("Hugging Face API Key")

prompt = st.text_input("Prompt here")

title_template = PromptTemplate(input_variables=["topic"],
                                template = "Tell me about {topic}")

repo_id = st.text_input("HF-Model")
title_chain = LLMChain(llm=HuggingFaceHub(repo_id=repo_id, 
                                    model_kwargs={'temperature':0.9,}),
                 prompt = title_template)


if st.button("Send"):
    response = title_chain.run(topic = prompt)
    st.write(response)

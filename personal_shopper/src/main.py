import streamlit as st
import requests
from PIL import Image
import langchain
import time
from langchain.chains import RetrievalQA
from langchain.document_loaders import GCSDirectoryLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import List
import uuid
import numpy as np
import json
import textwrap


import os
import urllib.request


class CustomVertexAIEmbeddings(VertexAIEmbeddings, BaseModel):
    requests_per_minute: int
    num_instances_per_batch: int

    def rate_limit(self, max_per_minute):
        period = 60 / max_per_minute
        print("Waiting")
        while True:
            before = time.time()
            yield
            after = time.time()
            elapsed = after - before
            sleep_time = max(0, period - elapsed)
            if sleep_time > 0:
                print(".", end="")
                time.sleep(sleep_time)
    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = self.rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]



def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def get_prompt(): 
    template = """SYSTEM: You are a laywer expert, you will receive question and a document context and need to answer user question based on the context provided
    Try to return if possible the article number where you took the info
Question: {question}


Strictly use the information you have in the context to answer and think step by step.
=============
Context: {context}
=============
    """

    return template




def get_retrieval_qa(template):
    from utils.matching_engine import MatchingEngine
    from utils.matching_engine_utils import MatchingEngineUtils

    mengine = MatchingEngineUtils('project-demo-389821', 'us-central1', 'projects/865280436803/locations/us-central1/indexes/2926064324302602240')
    EMBEDDING_QPM = 100
    EMBEDDING_NUM_BATCH = 5
    embeddings = CustomVertexAIEmbeddings(
        requests_per_minute=EMBEDDING_QPM,
        num_instances_per_batch=EMBEDDING_NUM_BATCH,
    )
    me = MatchingEngine.from_components(
            project_id='project-demo-389821',
            region='us-central1',
            gcs_bucket_name=f"gs://project-demo-389821-me-bucket1".split("/")[2],
            embedding=embeddings,
            index_id='projects/865280436803/locations/us-central1/indexes/2926064324302602240',
            endpoint_id='projects/865280436803/locations/us-central1/indexEndpoints/2101905592493801472',
        )
  

    retriever = me.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6,
        "search_distance": 0.6,
    },
    )

    llm = VertexAI(
        model_name="text-bison",
        max_output_tokens=2040,
        temperature=0.4,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=template,
                input_variables=["context", "question"],
            ),
        },
    )
    return qa

def ask(qa, query): 
    qa.retriever.search_kwargs["search_distance"] = 0.6
    qa.retriever.search_kwargs["k"] = 6
    qa.combine_documents_chain.verbose = True
    qa.combine_documents_chain.llm_chain.verbose = True
    qa.combine_documents_chain.llm_chain.llm.verbose = True
    result = qa({"query": query})
    return result['result']

def main():
    if not os.path.exists("utils"):
        os.makedirs("utils")

    url_prefix = "https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/use-cases/document-qa/utils"
    files = ["__init__.py", "matching_engine.py", "matching_engine_utils.py"]
    qa = get_retrieval_qa(get_prompt())

    for fname in files:
        urllib.request.urlretrieve(f"{url_prefix}/{fname}", filename=f"utils/{fname}")

    st.title("Arriaga asociados")
    st.markdown("#### Personal assistant")
    st.sidebar.image(add_logo(logo_path='./assets/arriaga.png', width=426, height=300)) 

    if "messages" not in st.session_state:
        st.chat_message("assistant").markdown("Hola, soy Willy tu gestor de documentos de Arriaga")
        st.session_state.messages = []


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Escribe tus consultas"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = f"{ask(qa, prompt)}"
        with st.chat_message("assistant"):
            st.markdown(response)


        st.session_state.messages.append({"role": "assistant", "content": response})




if __name__ == "__main__":
    main()

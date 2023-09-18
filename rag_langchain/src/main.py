import streamlit as st
from PIL import Image
import time
from langchain.chains import RetrievalQA
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List
import os
import urllib.request
import vertexai
import yaml

class AppConfig:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.load_config()

    def load_config(self):
        with open(self.yaml_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        self.project_id = yaml_data.get('project_id', None)
        self.region = yaml_data.get('region', None)
        self.img_path = yaml_data.get('image_path', None)
        self.index_id = yaml_data.get('index_id', None)
        self.endpoint_id = yaml_data.get('endpoint_id', None)
        self.prompt = yaml_data.get('prompt_palm_improvement', None)
        vertexai.init(project=self.project_id, location=self.region)


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
    def embed_documents(self, texts: List[str]):
        limiter = self.rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]



def add_logo(logo_path, width, height):
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo




def get_retrieval_qa(template, index_id, endpoint_id, project_id, region):
    from utils.matching_engine import MatchingEngine
    from utils.matching_engine_utils import MatchingEngineUtils

    mengine = MatchingEngineUtils(f'{project_id}', f'{region}', f'{index_id}')
    EMBEDDING_QPM = 100
    EMBEDDING_NUM_BATCH = 5
    embeddings = CustomVertexAIEmbeddings(
        requests_per_minute=EMBEDDING_QPM,
        num_instances_per_batch=EMBEDDING_NUM_BATCH,
    )
    me = MatchingEngine.from_components(
            project_id=f'{project_id}',
            region=f'{region}',
            gcs_bucket_name=f"gs://{project_id}-me-bucket1".split("/")[2],
            embedding=embeddings,
            index_id=index_id,
            endpoint_id=endpoint_id,
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

def main(config):
    if not os.path.exists("utils"):
        os.makedirs("utils")

    url_prefix = "https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/use-cases/document-qa/utils"
    files = ["__init__.py", "matching_engine.py", "matching_engine_utils.py"]
    qa = get_retrieval_qa(config.prompt, config.index_id, config.endpoint_id, config.project_id, config.region)

    for fname in files:
        urllib.request.urlretrieve(f"{url_prefix}/{fname}", filename=f"utils/{fname}")

    st.title("Firma de abogados")
    st.markdown("#### Asistente personal")
    st.sidebar.image(add_logo(logo_path=config.img_path, width=426, height=300)) 

    if "messages" not in st.session_state:
        st.chat_message("assistant").markdown("Hola, soy Willy tu gestor de documentos")
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
    config = AppConfig('config.yaml')
    main(config)

from google.cloud import aiplatform
import streamlit as st
import vertexai
from vertexai.language_models import TextGenerationModel
import re
import subprocess
import json
import base64
from PIL import Image
import io
import numpy as np 
import requests
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
        self.prompt = yaml_data.get('prompt_palm_improvement', None)
        self.url_img_gen = f'https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/imagegeneration:predict'.format(
            PROJECT_ID=self.project_id, REGION=self.region)
        vertexai.init(project=self.project_id, location=self.region)

        
            



def call_image_apis(url, payload):
    try:
        access_token = subprocess.check_output("gcloud auth print-access-token", shell=True, text=True).strip()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  
        return response.json()
    except subprocess.CalledProcessError as e:
        st.error(f"Error getting access token: {3}")
        return None  
    except requests.exceptions.RequestException as e:
        st.error(f"Error making the API request: {e} ")
        return None  # Handle the error accordingly
    except Exception as e:
        st.error(f"An unexpected error occurred {e}")
        return None  

def img_gen_payload(prompt):
    request_payload = {
        "instances": [
            {
                "prompt": "{}".format(prompt)
            }
        ],
        "parameters": {
            "sampleCount": 4
        }
    }
    return request_payload


def get_prompts(title, options, prompt):
    parameters = {
        "temperature": 0.4,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(f"{prompt.format(options, title)}",
        **parameters
    )
    pattern = r"\d+: (.*)"

    matches = re.findall(pattern, response.text)
    sentences = []
    for idx, sentence in enumerate(matches, 1):
        sentences.append(sentence)
    return sentences


def request_create_image(prompt, url):
        payload =  img_gen_payload(prompt)
        data = call_image_apis(url, payload)
        st.markdown("## Images")

        predictions = data.get("predictions", [])
        imglst = []
        if not predictions:
            st.warning("No predictions found in the JSON data.")
        else:
            for i, prediction in enumerate(predictions):
                mime_type = prediction.get("mimeType")
                encoded_image = prediction.get("bytesBase64Encoded")
                img = base64_to_pil_image(encoded_image)
                imglst.append(img)
        st.image(imglst)


def create_pics(prompt):
    col1, col2 = st.columns(2)
    with col1:
        prompt_field = st.text_input('Prompt to generate the image', prompt)

    with col2:
        st.text("")
        st.text("")
        button_img = st.button('Create Image')
    
def base64_to_pil_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = io.BytesIO(byte_data)
    img = Image.open(image_data)
    return img

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def main(config):
    sentences = ['', '', '']
    url = config.url_img_gen
    if 'button1_clicked' not in st.session_state:
        st.session_state['button1_clicked'] = False
    if 'button2_clicked' not in st.session_state:
        st.session_state['button2_clicked'] = False
    if 'button3_clicked' not in st.session_state:
        st.session_state['button3_clicked'] = False
    if 'button_clicked' not in st.session_state:
        st.session_state['button_clicked'] = False
    st.sidebar.image(add_logo(logo_path=config.img_path, width=426, height=300)) 
    st.title('Prompt Coach')


    st.markdown('## Generación de images custom')
    col1, col2, col3 = st.columns(3)
    with col1:
        title = st.text_input('Descripcion del contenido a mostrar', 'Woman with barbie dress')

    with col2:
        options = st.multiselect(
            'Tipo de imagen',
            ['4k', 'Drawing', 'Anime', 'Black and White Photo'], ['4k']
        )
    with col3:
        st.text("")
        st.text("")
        button = st.button('Submit')

    if button:
        st.session_state['button_clicked'] = True
        st.markdown('#### Sugerencias de prompts. Elige una')
        st.session_state['sentences'] = get_prompts(title, options[0], config.prompt)

        # Note: you may want to handle cases when sentences list is empty
    if 'sentences' in st.session_state:
        sent = st.session_state['sentences']
        button1 = st.button(sent[0])
        button2 = st.button(sent[1])
        button3 = st.button(sent[2])
        if button1:
            st.session_state['button1_clicked'] = not st.session_state['button1_clicked']
        if button2:
            st.session_state['button2_clicked'] = not st.session_state['button2_clicked']
        if button3:
            st.session_state['button3_clicked'] = not st.session_state['button1_clicked']

    if st.session_state['button1_clicked']:
        
       # st.session_state.disabled = True
        #create_pics(st.session_state['sentences'][0])
        request_create_image(st.session_state['sentences'][0], url)
    if st.session_state['button2_clicked']:
            # create_pics(st.session_state['sentences'][0])
        request_create_image(st.session_state['sentences'][1], url)
    if st.session_state['button3_clicked']:
                # create_pics(st.session_state['sentences'][0])
        request_create_image(st.session_state['sentences'][2], url)
     




if __name__ == '__main__':
    config = AppConfig('config.yaml')
    main(config)
    

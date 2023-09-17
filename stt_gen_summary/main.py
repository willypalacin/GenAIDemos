

from google.cloud import aiplatform
import streamlit as st
import vertexai
from vertexai.language_models import TextGenerationModel
import re
import subprocess
import vertexai
import json
import base64
from PIL import Image
import io
import numpy as np 
import cv2
import requests
from google.cloud import speech
import yaml
from tempfile import NamedTemporaryFile
import json
import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair




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
        self.url_img_cpt = "https://us-central1-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/imagetext:predict".format(self.project_id, self.region)
        vertexai.init(project=self.project_id, location=self.region)


    


def vision_api_call(image_bytes):
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.content = image_bytes
    response = client.label_detection(image=image)

    labels = response.label_annotations
    print("Labels:")
    label_descriptor = []
    object_descriptor = []
    for label in labels:
        label_descriptor.append(label.description)

    objects = client.object_localization(image=image).localized_object_annotations

    print(f"Number of objects found: {len(objects)}")
    for object_ in objects:
        object_descriptor.append(f"\n{object_.name} (confidence: {object_.score})")
    return label_descriptor, object_descriptor 


def bytes_to_base64(opencv_image):
    _, buffer = cv2.imencode('.jpg', opencv_image)
    base64_image = base64.b64encode(buffer).decode()
    return base64_image


                #st.image(encoded_image, caption="Base64 Encoded Image", use_column_width=True)


def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def call_palm_generate_verbatim(prompt, description):
    chat_model = ChatModel.from_pretrained("chat-bison")
    parameters = {
       # "candidate_count": 1,
        "max_output_tokens": 2000,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chat_model.start_chat(
        context=prompt.format(description))
    response = chat.send_message("Sumariza el contenido de este podcast", **parameters)
    st.markdown("## Resumen del podcast:")
    st.write(response.text)
    return chat

def img_captioning_payload(base64_image, lan):
    request_payload = {
            "instances": [
                {
                    "image": {
                        "bytesBase64Encoded": base64_image
                    }
                }
            ],
         "parameters": {
                "sampleCount": "4",
                "language": lan
            }
        }
    return request_payload

def transcribe_file_with_enhanced_model(path) -> speech.RecognizeResponse:
    """Transcribe the given audio file using an enhanced model."""

    client = speech.SpeechClient()
    with open(path, "rb") as audio_file:
        content = audio_file.read()
    

    # path = 'resources/commercial_mono.wav'
 

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="es-ES",
        use_enhanced=True,
        audio_channel_count=2,
       
        # A model must be specified to use enhanced model.
        model="phone_call",
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    st.info("Esperando a que complete transcripcion...")
    response = operation.result(timeout=90)
    #st.write(str(response) )

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        print("-" * 20)
        print(f"First alternative of result {i}")
        st.markdown(f"> Transcript: {alternative.transcript}")
        return alternative.transcript

    
def print_result(result: speech.SpeechRecognitionResult):
    best_alternative = result.alternatives[0]
    st.write("-" * 80)
    st.write(result)
    #print(f"language_code: {result.language_code}")
    #st.write(f"transcript:    {best_alternative.transcript}")

def print_response(response: speech.RecognizeResponse):
    for result in response.results:
        st.write(result)
        print_result(result)

    #print(f"confidence:    {best_alternative.confidence:.0%}")
def create_chat(chat):
    parameters = {
        #"candidate_count": 1,
        "max_output_tokens": 2000,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40
    }
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #for message in st.session_state.messages:
     #   with st.chat_message(message["role"]):
      #      st.markdown(message["content"])

    if prompt := st.chat_input("Hola soy tu chat en base a podcast"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})


        response = chat.send_message(prompt, **parameters)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.text)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def main(config):
    
    st.sidebar.image(add_logo(logo_path=config.img_path, width=426, height=300)) 
    st.title('Generative AI - Prensa Ibérica')

    st.markdown("## Generar sumarizacion de audio")
    uploaded_file = st.file_uploader("Upload a File", type=["mp3", "m4a", "wav", "aiff"])
    if uploaded_file is not None:
        with NamedTemporaryFile(suffix="wav") as temp:
            temp.write(uploaded_file.getvalue())
            temp.seek(0)
            st.audio(temp.read())
            # To read file as bytes:
            response = transcribe_file_with_enhanced_model(temp.name)
            #st.write(response)
            #print_response(response)
            chat = call_palm_generate_verbatim(config.prompt, response)
            create_chat(chat)
        #call_palm_generate_verbatim(str(preds), str(labels), str(objects), config.prompt)



        

if __name__ == '__main__':
    config = AppConfig('config.yaml')
    main(config)
    #st.text("Made by gpalacin@")



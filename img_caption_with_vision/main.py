

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
from google.cloud import vision
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

def call_palm_generate_verbatim(description, labels, objects, prompt):
    parameters = {
        "temperature": 0.4,
        "max_output_tokens": 2000,
        "top_p": 0.8,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(
        f"{prompt.format(description, labels, objects)}",
        **parameters
    )
    # Output the extracted sentences
    st.markdown("### ——————PALM——————")
    st.write(response.text)

def img_captioning_payload_qa(base64_image, prompt):
    request_payload = {
            "instances": [
                {
                    "prompt": prompt,
                    "image": {
                        "bytesBase64Encoded": base64_image
                    }
                }
            ],
         "parameters": {
                "sampleCount": "2",
            }
        }
    return request_payload

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


def main(config):
    
    st.sidebar.image(add_logo(logo_path=config.img_path, width=426, height=300)) 
    st.title('Generative AI - Use Cases')

    st.markdown("## Generar descripciones a partir de imagenes")
    uploaded_file = st.file_uploader("Upload a File", type=["jpg", "png"])
    if uploaded_file is not None:
        # To read file as bytes:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")
        base64_image = bytes_to_base64(opencv_image)
        #st.write("——————VISION API ATTR——————")

        labels, objects = vision_api_call(uploaded_file.getvalue())
        #st.write(labels)
        #st.write(objects)
        url_img_cpt = config.url_img_cpt
        response_en = call_image_apis(url_img_cpt, img_captioning_payload(base64_image, "en"))
        preds = []
        if response_en is not None: 
            #st.write("——————IMAGE CAPTIONING——————")

            for pred in response_en['predictions']:
                preds.append(pred)
                #st.write(pred)

        
        response_qa = call_image_apis(url_img_cpt, img_captioning_payload_qa(base64_image, 'type of trousers'))
        if response_qa is not None: 
            for pred in response_qa['predictions']:
                preds.append("type of trousers: " + pred)
   #             st.write(pred)

        response_qa = call_image_apis(url_img_cpt, img_captioning_payload_qa(base64_image, 'type and color of top part clothing'))
        if response_qa is not None: 
            for pred in response_qa['predictions']:
                preds.append("type of upper part: " + pred)
        
        call_palm_generate_verbatim(str(preds), str(labels), str(objects), config.prompt)
  

if __name__ == '__main__':
    config = AppConfig('config.yaml')
    main(config)



# GenAI Labs
`by gpalacin@`

This repository contains a series of isolated labs related to generative AI on the Google Cloud Platform. 

These quick prototypes showcase the different generative AI capabilities available in Vertex AI, with real-world use cases.

> Note: Most of these demo contant is in Spanish.

## Pre-requisites
Before starting you will need: 
1. A Google Cloud Project with a biilling account associated 
2. Shell environment with gcloud and git

## Set up
```bash
gcloud auth login
```
```bash
git clone https://github.com/willypalacin/GenAIDemos
``````

Enable the following APIs
```bash 
gcloud services enable \
artifactregistry.googleapis.com \
vertexai.googleapis.com \
cloudbuild.googleapis.com \
cloudrun.googleapis.com \
compute.googleapis.com
```
All the demos will be deployed in Cloud Run. We need to create the container to store them.

```bash 
gcloud artifacts repositories create gen-ai-imgs \
--repository-format=docker \
--location=$REGION --description="Registry to store the demo images"
```


## Demo 1 - Prompt Coach
Given a particular prompt, refine it with PaLM to coah the user and generate a high quality image.







<p align="center">
  <img src="./imgs/mm1.gif" alt="Image Description" />
</p>

![Architecture](./imgs/arch1.png)
Build and push the image to artifact registry


```
cd custom_img_gen

gcloud builds submit -t $REGION-docker.pkg.dev/$PROJECT_ID/gen-ai-imgs/img-caption . 
```

Deploy to Cloud Run

```
gcloud run deploy img-gen-service \
--image="${REGION}-docker.pkg.dev/${PROJECT_ID}/gen-ai-imgs/img-caption" \
--region=$REGION \
--allow-unauthenticated
```

# Demo 2 - Image descriptor
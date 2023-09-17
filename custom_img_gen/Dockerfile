FROM python:3.9

COPY . /app

WORKDIR /app
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean

# Download and install the Google Cloud SDK
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-446.0.1-linux-x86_64.tar.gz && \
    tar -xzf google-cloud-sdk-446.0.1-linux-x86_64.tar.gz -C /opt && \
    rm google-cloud-sdk-446.0.1-linux-x86_64.tar.gz

# Add gcloud command to PATH
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

RUN pip install -r requirements.txt


CMD ["streamlit", "run", "main.py", "--server.port", "8080"]
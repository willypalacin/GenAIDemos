FROM python:3.9

COPY . /app

WORKDIR /app
RUN apt-get update && \
    apt-get install -y curl libgl1-mesa-glx && \
    apt-get clean

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-446.0.1-linux-x86_64.tar.gz && \
    tar -xzf google-cloud-sdk-446.0.1-linux-x86_64.tar.gz -C /opt && \
    rm google-cloud-sdk-446.0.1-linux-x86_64.tar.gz

ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

RUN pip install -r requirements.txt


CMD ["streamlit", "run", "src/main.py", "--server.port", "8080"]
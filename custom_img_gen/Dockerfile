FROM python:3.9

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "main.py", "--server.port", "8080"]
FROM python:3.9.21-slim

WORKDIR /app/LocalRAG

COPY ./LocalRAG/ .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
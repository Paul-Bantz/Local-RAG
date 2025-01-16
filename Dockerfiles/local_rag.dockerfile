FROM nvidia/cuda:12.6.3-base-ubuntu24.04

RUN apt-get update \
 && apt-get install -y python3-pip \
 && rm /usr/lib/python*/EXTERNALLY-MANAGED

WORKDIR /app/LocalRAG
COPY ./LocalRAG/ requirements.txt .

RUN pip3 install -r requirements.txt "gpt4all[cuda]"

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
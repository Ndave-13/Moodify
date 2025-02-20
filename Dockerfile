FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir tensorflow spotipy streamlit numpy opencv-python

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

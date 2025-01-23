from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64

app = FastAPI()

# Load the trained model
model_best = load_model('face_model.h5')
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Spotify API setup
SPOTIPY_CLIENT_ID = "08240ef15cf743c0a3784b440e4250e5"
SPOTIPY_CLIENT_SECRET = "da9300425a6f460fb2e8678d03a78a79"
client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_spotify_recommendations(mood):
    """
    Fetch Spotify recommendations based on detected mood.
    """
    try:
        results = sp.search(q=mood, type="track", limit=5)
        tracks = [
            {
                "name": track["name"],
                "artist": ", ".join(artist["name"] for artist in track["artists"]),
                "url": track["external_urls"]["spotify"],
            }
            for track in results["tracks"]["items"]
        ]
        return tracks
    except Exception as e:
        return []


@app.get("/", response_class=HTMLResponse)
async def index():
    return templates.TemplateResponse("index.html", {"request": {}})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_text("Error: Could not open the webcam.")
        return

    emotion_label = None
    while True:
        ret, frame = cap.read()
        if not ret or emotion_label:
            break  # Stop capturing frames once emotion is detected

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_image = cv2.resize(face_roi, (48, 48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = image.img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)

            predictions = model_best.predict(face_image)
            emotion_label = class_names[np.argmax(predictions)]

            # Draw rectangle and text on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        await websocket.send_json({"image": jpg_as_text, "emotion": emotion_label or "Detecting..."})

    cap.release()

    # Wait for recommendation request
    while True:
        data = await websocket.receive_json()
        if data.get("action") == "generate_recommendations" and emotion_label:
            recommendations = get_spotify_recommendations(emotion_label)
            await websocket.send_json({"recommendations": recommendations})
            break

    await websocket.close()

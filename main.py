import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
import random
import streamlit.components.v1 as components

# Load the trained emotion detection model
model_best = load_model('face_model.h5')
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Spotify API setup (Replace with your credentials)
SPOTIPY_CLIENT_ID = "b66c9baebce5453081456d9d994a0c34"
SPOTIPY_CLIENT_SECRET = "0fa6bd50f09646279c94fcdc297686ba"

client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Map emotions to playlist categories
mood_to_category = {
    "Happy": "party",
    "Sad": "sad",
    "Angry": "rock",
    "Fear": "focus",
    "Surprise": "pop",
    "Neutral": "chill",
    "Disgusted": "metal",
}

def get_spotify_recommendations_by_category(category):
    """
    Fetch random Spotify songs based on mood-related playlists.
    """
    try:
        results = sp.search(q=category, type="playlist", limit=5)

        if not results or "playlists" not in results or "items" not in results["playlists"]:
            st.error("No playlists found for this category.")
            return []

        playlists = results["playlists"]["items"]
        if not playlists:
            st.error("No playlists available.")
            return []

        playlist = random.choice(playlists)  # Pick a random playlist
        playlist_id = playlist["id"]
        tracks_data = sp.playlist_tracks(playlist_id, limit=10)

        if not tracks_data or "items" not in tracks_data:
            st.error("No tracks found in the selected playlist.")
            return []

        tracks = tracks_data["items"]
        random.shuffle(tracks)  # Shuffle for randomness

        recommendations = []
        for track in tracks:
            if "track" in track and track["track"] and "external_urls" in track["track"]:
                recommendations.append({
                    "name": track["track"]["name"],
                    "artist": ", ".join(artist["name"] for artist in track["track"]["artists"]),
                    "url": track["track"]["external_urls"]["spotify"],
                })
            if len(recommendations) >= 5:
                break

        if not recommendations:
            st.error("No valid songs found. Try again!")
        
        return recommendations

    except Exception as e:
        st.error(f"Error fetching songs from Spotify: {str(e)}")
        return []

def song_page(song_url, song_name, artist_name):
    """Displays the embedded Spotify song for each recommended track."""
    st.subheader(f"🎵 Now Playing: {song_name} - {artist_name}")
    st.markdown('---')
    song_uri = song_url.split('/')[-1].split('?')[0]
    uri_link = f'https://open.spotify.com/embed/track/{song_uri}'
    components.iframe(uri_link, height=100)

# Streamlit UI
st.title("🎭 Mood-Based Music Recommendation 🎵")
st.write("This app detects your mood using your webcam and suggests music based on your emotions.")

emotion_label = None

frame_placeholder = st.empty()
status_placeholder = st.empty()

detect_button = st.button("Start Emotion Detection")

if detect_button:
    st.write("📸 Capturing your face and detecting emotion...")
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not cap.isOpened():
        st.error("Error: Could not open the webcam.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Unable to capture image from webcam.")
                break

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

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_column_width=True)

            if emotion_label:
                status_placeholder.success(f"Detected Emotion: {emotion_label}")
                time.sleep(2)
                break
        
        cap.release()
        cv2.destroyAllWindows()

if emotion_label:
    category = mood_to_category.get(emotion_label, "pop")
    st.subheader(f"🎶 Recommended Songs for Your Mood ({emotion_label})")
    recommendations = get_spotify_recommendations_by_category(category)

    if recommendations:
        for track in recommendations:
            st.write(f"🎵 [{track['name']} - {track['artist']}]({track['url']})")
            song_page(track['url'], track['name'], track['artist'])  # Show embedded player for each song
    else:
        st.error("No recommendations found. Try again!")

st.write("Press the Start Emotion Detection button to begin!")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from keras.models import load_model
import numpy as np
import shutil
import os
import cv2

# Initialize FastAPI app
app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = load_model("final_lstm_model.keras")

# Labels based on your training classes (adjust if needed)
class_labels = ["Abuse", "Arrest", "Arson", "Assault", "Burglary",
                "Explosion", "Fighting", "Normal", "RoadAccident",
                "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

# Directory to temporarily save videos
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper function to extract and preprocess frames
def extract_frames(video_path, size=(128, 128), max_frames=128):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale
        frame = frame.astype("float32") / 255.0
        frame = frame.reshape(-1)[:384]  # flatten to 384 values
        frames.append(frame)

    cap.release()

    while len(frames) < max_frames:
        frames.append(np.zeros(384, dtype=np.float32))

    return np.array(frames[:max_frames])  # (128, 384)

# Root route
@app.get("/")
def root():
    return {"message": "AI CCTV Anomaly Detection API is running"}

# Prediction endpoint
@app.post("/predict/")
async def predict(video: UploadFile = File(...)):
    try:
        # Save video
        file_path = os.path.join(UPLOAD_DIR, video.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Extract frames
        frames = extract_frames(file_path)
        input_data = np.expand_dims(frames, axis=0)  # (1, 128, 384)

        # Predict
        predictions = model.predict(input_data)[0]
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        result = class_labels[predicted_class]

        return JSONResponse(content={
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

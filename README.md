🧠 AI-Based CCTV Anomaly Detection System
An intelligent deep learning system for detecting abnormal human activities from CCTV footage using CNN-LSTM architecture.
📌 Overview

The AI CCTV Anomaly Detection System is a deep learning-based application designed to automatically identify suspicious and abnormal activities in surveillance videos.
The system combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal sequence learning.

This project aims to assist human surveillance operators by providing automatic alerts for activities such as:

Fighting

Theft

Arson

Wall jumping / Trespassing

Accidents

It uses the UCF-Crime Dataset as the source of real-world surveillance videos.

🎯 Project Objectives

Automate the detection of anomalies in CCTV footage

Reduce human workload and improve accuracy of surveillance systems

Develop an efficient deep learning model using CNN-LSTM

Build a user-friendly web interface for video upload and anomaly detection

🧩 System Architecture
+-----------------+      +------------------+      +----------------+      +-------------------+
|  CCTV Footage   | -->  |  Frame Extraction | --> |  CNN Feature   | -->  |  LSTM Anomaly     |
|  (Input Video)  |      |  & Preprocessing  |     |  Extraction    |     |  Classification   |
+-----------------+      +------------------+      +----------------+      +-------------------+
                                                                             |
                                                                             v
                                                                +-------------------------+
                                                                |  Alert / Detection Result|
                                                                +-------------------------+

🧠 Model Details

Architecture: CNN + LSTM

Input Size: 224 × 224 frames

Frame Rate: 1 FPS

Feature Extractor: VGG16 (pretrained on ImageNet)

Sequence Length: 20 frames per clip

Optimizer: Adam

Loss Function: Categorical Crossentropy

Accuracy Achieved: ~87.6%

🗂️ Dataset Used

Dataset: UCF-Crime Dataset

Real-world surveillance videos

13 anomaly classes + normal class

Classes used in this project:
Fighting, Theft, Arson, Accident, Trespassing (Wall Jumping), and Normal

⚙️ Preprocessing Steps

Extract frames from videos at 1 FPS

Resize all frames to 224×224 pixels

Normalize pixel values to the range [0, 1]

Convert frames into sequential clips (e.g., 20 frames per sequence)

Encode labels for supervised training

Save as .npy files or use ImageDataGenerator

💻 Technologies Used
Category	Tools / Libraries
Programming Language	Python
Deep Learning	TensorFlow, Keras
Video Processing	OpenCV
Data Handling	NumPy, Pandas
Model Serving	FastAPI
Frontend	HTML, CSS, JavaScript
Environment	Google Colab (TPU/GPU)
Storage	Google Drive (2TB student offer)
🧰 Project Setup
1. Clone the Repository
git clone https://github.com/<your-username>/AI-CCTV-Anomaly-Detection.git
cd AI-CCTV-Anomaly-Detection

2. Install Dependencies
pip install -r requirements.txt

3. Run Preprocessing
python preprocessing/extract_frames.py

4. Train the Model
python train_model.py

5. Run Backend (FastAPI)
uvicorn main:app --reload

6. Open Frontend

Use your browser to visit:

http://127.0.0.1:8000

📈 Results
Metric	Score
Accuracy	87.6%
Precision	85.2%
Recall	84.7%
F1-Score	84.9%

✅ The model effectively classifies anomalies and distinguishes between normal and abnormal behaviors.
⚠️ Slight misclassifications occur in low-light or crowded scenes.

🚀 Future Enhancements

Real-time video feed integration using OpenCV

Add more anomaly categories

Implement multi-camera tracking

Build an alert notification system (email/SMS)

Optimize model for edge devices (Jetson Nano, Raspberry Pi)

📜 Project Structure
AI-CCTV-Anomaly-Detection/
├── preprocessing/
│   ├── extract_frames.py
│   ├── create_sequences.py
├── model/
│   ├── cnn_lstm_model.py
│   ├── saved_model.h5
├── backend/
│   ├── main.py
│   ├── utils.py
├── frontend/
│   ├── index.html
│   ├── styles.css
├── notebooks/
│   ├── training_colab.ipynb
├── README.md
└── requirements.txt

🧾 License

This project is licensed under the MIT License – feel free to use and modify it for research and educational purposes.

👨‍💻 Contributors

N. Dravinesh – AI Engineer & Developer

Department of Data science and Engineering

Dr.MGR educational and research institute
⭐ Support

If you find this project useful, please ⭐ the repository on GitHub and share it!
For suggestions or collaboration opportunities, feel free to connect.

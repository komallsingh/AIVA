# 🎙️🧠 AIVA – AI Virtual Assistant  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyAudio](https://img.shields.io/badge/PyAudio-008080?style=for-the-badge)
![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-FF9800?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-6A5ACD?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

> 🚀 AIVA is a multimodal AI assistant that processes **face and speech input** to understand user emotion and generate intelligent responses in real time.

---

## 📌 Overview

**AIVA (AI Virtual Assistant)** is an AI-based system that combines:

- 🎥 Computer Vision (Face Detection)  
- 🎤 Speech Recognition (Audio → Text)  
- 🧠 NLP (Text Understanding + Response Generation)  
- 🖥️ Interactive UI (Streamlit)  

It enables **real-time human-computer interaction** using both visual and voice inputs.

---

## ✨ Features

- 🎥 Real-time face detection using webcam  
- 🎤 Speech-to-text conversion  
- 🧠 NLP-based response generation  
- 💬 Intelligent response engine  
- ⚡ Fast and lightweight processing  
- 🖥️ Interactive Streamlit interface  

---

## 🧠 Tech Stack

| ⚙️ Category | 🚀 Technology |
|------------|-------------|
| Language | Python |
| Computer Vision | OpenCV (Haarcascade) |
| Audio Input | PyAudio |
| Speech Recognition | SpeechRecognition |
| NLP | Basic NLP techniques |
| UI Framework | Streamlit |

---

## ⚙️ How It Works

### 🎥 Face Detection
- Implemented in `face_detect.py`  
- Uses Haarcascade with OpenCV  
- Detects faces in real-time from webcam  

---

### 🎤 Audio Emotion / Speech Processing
- Implemented in `audio_emotion.py`  
- Captures audio using PyAudio  
- Converts speech to text using SpeechRecognition  

---

### 🧠 NLP & Response Engine
- Implemented in `response_engine.py`  
- Processes text input  
- Generates intelligent responses based on user speech  

---

### 🖥️ Main Application
- `app.py` integrates:
  - Face detection  
  - Speech processing  
  - NLP response  
- Provides an interactive UI using Streamlit  

---

## 🎯 Output

- 🎥 Detected face (visual confirmation)  
- 🗣️ Recognized speech (text)  
- 💬 AI-generated response  
- ⚡ Real-time interaction  

---

## 📊 Project Structure

```bash
AIVA/
│
├── model/                 # Stored ML/NLP models
│
├── app.py                 # Main Streamlit application
├── face_detect.py         # Face detection module
├── audio_emotion.py       # Audio processing & speech recognition
├── response_engine.py     # NLP response logic
│
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── .gitignore             # Ignored files
```

---
## 👩‍💻 Authors

**Komal Singh**
[![GitHub](https://img.shields.io/badge/GitHub-komallsingh-181717?style=flat&logo=github&logoColor=white)](https://github.com/komallsingh)

**Pranjal Singh**
[![GitHub](https://img.shields.io/badge/GitHub-Pranjalsingh10-181717?style=flat&logo=github&logoColor=white)](https://github.com/Pranjalsingh10)

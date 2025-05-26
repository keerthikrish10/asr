🧠 NeMo ASR FastAPI App
A production-ready FastAPI-based automatic speech recognition (ASR) app using NVIDIA NeMo models, with Docker support and preprocessing customization.

📌 Table of Contents
About the Project

Built With

Getting Started

Prerequisites

Installation

Usage

Roadmap

Contributing

License

Contact

📖 About The Project
This project provides an API to transcribe short audio clips (1–30 seconds) into text using NVIDIA NeMo's advanced speech recognition models.

Key Features:

Audio format validation

MP3 to WAV auto-conversion

Preprocessing controls (low, medium, high)

Background task cleanup of temp files

🏗️ Built With
FastAPI

Docker

NVIDIA NeMo

Uvicorn

Librosa

FFmpeg

🚀 Getting Started
To get a local copy up and running follow these simple steps.

✅ Prerequisites
Docker installed on your system

FFmpeg installed (if running without Docker)

⚙️ Installation
Clone the repo

bash
Copy
Edit
git clone https://github.com/your_username/nemo-asr-fastapi.git
cd nemo-asr-fastapi
Build the Docker image

bash
Copy
Edit
docker build -t nemo-asr-app .
Run the container

bash
Copy
Edit
docker run -it --rm -p 8000:8000 nemo-asr-app
Open the API docs
Visit http://localhost:8000/docs

🧠 Usage
Use curl to hit the /transcribe endpoint:

🎯 Sample Request (Windows CMD)
bash
Copy
Edit
curl -X POST "http://localhost:8000/transcribe" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "audio_file=@C:/Users/KEERTHI%20KRISHANA/Downloads/Hindi_F_Deepika.mp3" ^
  -F "preprocessing_intensity=medium"
🔈 Ensure the audio duration is between 1 and 30 seconds.

🧩 Roadmap
 Add language selection support

 Add streaming support via WebSocket

 Deploy on HuggingFace or GCP

 Add client-side UI for file uploads

🤝 Contributing
Please refer to CONTRIBUTING.md for contribution guidelines.

📄 License
Distributed under the MIT License. See LICENSE.txt for details.

📬 Contact
Keerthi Krishna
📧 keerthikrishna@example.com
🔗 GitHub Repo


# 🧠 NeMo ASR FastAPI App

<div align="center">
  <a href="https://github.com/your_username/nemo-asr-fastapi">
    <img src="https://img.shields.io/badge/NVIDIA-NeMo-green?style=for-the-badge&logo=nvidia" alt="NeMo Logo">
  </a>

  <h3 align="center">🎤 Production-Ready Speech Recognition API</h3>

  <p align="center">
    A powerful FastAPI-based automatic speech recognition (ASR) application using NVIDIA NeMo models
    <br />
    <a href="https://github.com/your_username/nemo-asr-fastapi"><strong>📚 Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/nemo-asr-fastapi">🎯 View Demo</a>
    ·
    <a href="https://github.com/your_username/nemo-asr-fastapi/issues">🐛 Report Bug</a>
    ·
    <a href="https://github.com/your_username/nemo-asr-fastapi/issues">💡 Request Feature</a>
  </p>
</div>

<!-- BADGES -->
<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/github/license/your_username/nemo-asr-fastapi.svg)
![Issues](https://img.shields.io/github/issues/your_username/nemo-asr-fastapi.svg)
![Stars](https://img.shields.io/github/stars/your_username/nemo-asr-fastapi.svg)

</div>

---

## 📋 **Table of Contents**

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">🎯 About The Project</a>
      <ul>
        <li><a href="#built-with">🏗️ Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">🚀 Getting Started</a>
      <ul>
        <li><a href="#prerequisites">✅ Prerequisites</a></li>
        <li><a href="#installation">⚙️ Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">🧠 Usage</a></li>
    <li><a href="#roadmap">🗺️ Roadmap</a></li>
    <li><a href="#contributing">🤝 Contributing</a></li>
    <li><a href="#license">📄 License</a></li>
    <li><a href="#contact">📬 Contact</a></li>
  </ol>
</details>

---

## 🎯 **About The Project**

<div align="center">
  <img src="https://img.shields.io/badge/Audio%20Processing-1--30%20seconds-brightgreen?style=for-the-badge" alt="Audio Duration">
  <img src="https://img.shields.io/badge/Format%20Support-MP3%20%7C%20WAV-blue?style=for-the-badge" alt="Format Support">
  <img src="https://img.shields.io/badge/Preprocessing-3%20Levels-orange?style=for-the-badge" alt="Preprocessing">
</div>

This project provides a **high-performance API** to transcribe short audio clips (1–16.7 seconds) into text using NVIDIA NeMo's state-of-the-art speech recognition models. Built with production environments in mind, it offers robust audio processing capabilities with Docker containerization.

### ✨ **Key Features**

| Feature | Description |
|---------|-------------|
| 🎵 **Audio Validation** | Intelligent format validation and duration checking |
| 🔄 **Auto-Conversion** | Seamless MP3 to WAV conversion using FFmpeg |
| ⚙️ **Preprocessing Levels** | Customizable preprocessing (Low/Medium/High) |
| 🧹 **Smart Cleanup** | Background task cleanup of temporary files |
| 🐳 **Docker Ready** | Production-ready containerization |
| 📊 **API Documentation** | Interactive Swagger UI documentation |

---

## 🏗️ **Built With**

This section showcases the major frameworks and libraries that power this project:

<div align="center">

| Technology | Purpose | Badge |
|------------|---------|-------|
| **FastAPI** | Web Framework | ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white) |
| **Docker** | Containerization | ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white) |
| **NVIDIA NeMo** | Speech Recognition | ![NVIDIA](https://img.shields.io/badge/NVIDIA-NeMo-76B900?style=flat&logo=nvidia) |
| **Uvicorn** | ASGI Server | ![Uvicorn](https://img.shields.io/badge/Uvicorn-4051B5?style=flat) |
| **Librosa** | Audio Processing | ![Python](https://img.shields.io/badge/Librosa-3776AB?style=flat&logo=python&logoColor=white) |
| **FFmpeg** | Media Processing | ![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=flat&logo=ffmpeg&logoColor=white) |

</div>

---

## 🚀 **Getting Started**

Get your local copy up and running in just a few simple steps!

### ✅ **Prerequisites**

Before you begin, ensure you have the following installed:

- 🐳 **Docker** - [Install Docker](https://docs.docker.com/get-docker/)
- 🎬 **FFmpeg** (if running without Docker) - [Install FFmpeg](https://ffmpeg.org/download.html)

### ⚙️ **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/your_username/nemo-asr-fastapi.git
   cd nemo-asr-fastapi
   ```

2. **Build the Docker image**
   ```bash
   docker build -t nemo-asr-app .
   ```

3. **Run the container**
   ```bash
   docker run -it --rm -p 8000:8000 nemo-asr-app
   ```

4. **🎉 Access the API**
   
   Open your browser and navigate to: **http://localhost:8000/docs**

---

## 🧠 **Usage**

### 🎯 **API Endpoint**

Use the `/transcribe` endpoint to convert audio to text:

#### **Sample Request (Windows CMD)**
```bash
curl -X POST "http://localhost:8000/transcribe" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "audio_file=@C:/Users/KEERTHI%20KRISHANA/Downloads/Hindi_F_Deepika.mp3" ^
  -F "preprocessing_intensity=medium"
```

#### **Sample Request (Linux/macOS)**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@/path/to/your/audio.mp3" \
  -F "preprocessing_intensity=medium"
```

### 📊 **Preprocessing Levels**

| Level | Description | Use Case |
|-------|-------------|----------|
| **Low** | Minimal processing | Clean, high-quality audio |
| **Medium** | Balanced processing | General-purpose transcription |
| **High** | Maximum processing | Noisy or low-quality audio |

> ⚠️ **Important:** Ensure audio duration is between **1-30 seconds** for optimal performance.

---

## 🗺️ **Roadmap**

Our development roadmap for upcoming features:

- [x] ✅ Core ASR functionality
- [x] ✅ Docker containerization
- [x] ✅ Preprocessing customization
- [ ] 🌐 **Language selection support**
- [ ] 📡 **WebSocket streaming support**
- [ ] ☁️ **Cloud deployment (HuggingFace/GCP)**
- [ ] 🖥️ **Client-side UI for file uploads**
- [ ] 📊 **Batch processing support**
- [ ] 🔊 **Real-time audio processing**

See the [open issues](https://github.com/your_username/nemo-asr-fastapi/issues) for a full list of proposed features and known issues.

---

## 🤝 **Contributing**

Contributions make the open source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

### **How to Contribute**

1. **Fork** the Project
2. **Create** your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your Changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the Branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

For detailed contribution guidelines, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 **License**

Distributed under the **MIT License**. See [LICENSE.txt](LICENSE.txt) for more information.

---

## 📬 **Contact**

<div align="center">

**Keerthi Krishna**

[![Email](https://img.shields.io/badge/Email-keerthikrishna@example.com-red?style=for-the-badge&logo=gmail&logoColor=white)](mailto:skeerthi.krish@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-your_username-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/keerthikrish10)
[![Project Link](https://img.shields.io/badge/Project-nemo--asr--fastapi-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/keerthikrish10/asr/)

</div>

---

<div align="center">
  <p><strong>⭐ Don't forget to give the project a star if you found it helpful!</strong></p>
  
  ![Star History Chart](https://api.star-history.com/svg?repos=your_username/nemo-asr-fastapi&type=Date)
</div>

# 🚦 Traffic Analytics MLOps Pipeline

## 📌 Overview

This project demonstrates an end-to-end **MLOps pipeline for traffic analytics using video data**.
It covers the full lifecycle from data ingestion → model training → experiment tracking → deployment.

The goal is to simulate a **real-world production pipeline** for intelligent traffic monitoring systems.

---

## ⚙️ Features

* 🎥 Video-based object detection (vehicles)
* 🧠 Model training using deep learning (YOLO-based)
* 📊 Experiment tracking with MLflow
* 📦 Dockerized application for reproducibility
* 🔁 CI/CD pipeline with GitHub Actions
* 🚀 FastAPI-based inference service

---

## 🏗️ Project Structure

```
traffic-analytics-mlops/
│
├── data/                # Input data (videos not tracked in Git)
├── models/              # Saved models / weights
├── mlruns/              # MLflow experiment tracking
├── src/                 # Core source code
├── app/                 # FastAPI app for inference
├── Dockerfile           # Containerization
├── requirements.txt     # Dependencies
└── .github/workflows/   # CI/CD pipelines
```

---

## 📊 Data

⚠️ Due to size constraints, video data is **not stored in this repository**.

You can:

* Add your own traffic videos inside:

```
data/videos/
```

---

## 🧪 Training

Run model training:

```bash
python src/train.py
```

MLflow UI:

```bash
mlflow ui
```

Then open:
👉 http://localhost:5000

---

## 🔍 Inference

Run the FastAPI app:

```bash
uvicorn app.main:app --reload
```

API will be available at:
👉 http://127.0.0.1:8000

---

## 🐳 Docker

Build and run the container:

```bash
docker build -t traffic-ml .
docker run -p 8000:8000 traffic-ml
```

---

## 🔁 CI/CD

GitHub Actions pipeline:

* installs dependencies
* runs checks
* builds project

---

## 📈 Results

* Model detects vehicles in traffic videos
* Outputs annotated video / predictions
* Experiment metrics tracked via MLflow

---

## 🧠 Tech Stack

* Python
* PyTorch / Ultralytics YOLO
* MLflow
* FastAPI
* Docker
* GitHub Actions

---

## 🚀 Future Improvements

* Multi-object tracking (DeepSORT)
* Real-time streaming inference
* Deployment on cloud (AWS / Azure)
* Data versioning with DVC

---

## 👤 Author

Deepak L
Machine Learning | MLOps | Computer Vision

---


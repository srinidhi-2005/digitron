# MNIST Handwritten Digit Recognition (NumPy + FastAPI + Docker + CI/CD)

## Overview

This project implements a **handwritten digit recognition system** using the MNIST dataset.
A neural network is built **from scratch using NumPy (without deep learning frameworks)** and deployed as a **REST API using FastAPI**.
The application is fully **containerized with Docker** and integrated with a **CI/CD pipeline using GitHub Actions** for automated builds and validation.

## 🚀 Features

* Neural Network implemented from scratch using NumPy
* Trained on MNIST dataset (28×28 grayscale images)
* Achieves ~97% accuracy on test data
* REST API built using FastAPI
* Interactive API documentation via Swagger (`/docs`)
* Dockerized for consistent and portable deployment
* CI/CD pipeline using GitHub Actions for automation

## 🧠 Model Details

* Input Layer: 784 neurons (28×28 image)
* Hidden Layer: 128 neurons (ReLU activation)
* Output Layer: 10 neurons (Softmax)
* Loss Function: Cross-Entropy
* Optimization: Gradient Descent (manual backpropagation)

## 📁 Project Structure

```bash
handwritten_digit_recognizer/
│
├── MNIST/
│   └── raw/
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte.gz
│       ├── t10k-images-idx3-ubyte.gz
│       ├── t10k-labels-idx1-ubyte.gz
│
├── model.py                # Neural network implementation
├── train.py                # Training + evaluation + visualization
├── predict.py              # Model loading + inference
├── app.py                  # FastAPI backend
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── model.pkl               # Saved trained model
│
├── .github/
│   └── workflows/
│       └── docker.yml      # CI/CD pipeline
```

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model

```bash
python train.py
```

This generates:

```bash
model.pkl
```

## ▶️ Running the Application

### Run FastAPI Server

```bash
uvicorn app:app --reload
```

### Open API Docs

```
http://127.0.0.1:8000/docs
```

## 🧪 API Usage

### Endpoint: `/predict`

**Method:** POST

**Request Body:**

```json
{
  "input": [784 pixel values]
}
```

**Response:**

```json
{
  "prediction": 7
}
```

## 🐳 Docker Setup

### Build Docker Image

```bash
docker build -t mnist-app .
```

### Run Container

```bash
docker run -p 8000:8000 mnist-app
```

Access:

```
http://localhost:8000/docs
```

## 🔄 CI/CD Pipeline (GitHub Actions)

This project uses **GitHub Actions** to automate the build and validation process.

### Workflow File

```
.github/workflows/docker.yml
```

### What the Pipeline Does

* ✅ Triggers on every push to `main`
* ✅ Checks out repository code
* ✅ Installs project dependencies
* ✅ Validates application imports
* ✅ Builds Docker image
* ❌ Fails automatically if any step breaks

### Sample Workflow Configuration

```yaml
name: CI - MNIST Docker App

on:
  push:
    branches: [ "main" ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - run: pip install -r requirements.txt

    - run: python -c "import app"

    - run: docker build -t mnist-app .
```

## 📊 Results

* Test Accuracy: ~97%
* Efficient inference via API
* Visualized predictions using matplotlib

## 💼 Tech Stack

* Python
* NumPy
* FastAPI
* Uvicorn
* Docker
* Git & GitHub Actions

## 🚀 Future Improvements

* Add image upload (instead of raw pixel input)
* Build frontend UI for drawing digits
* Deploy to cloud platforms (AWS / Render)
* Upgrade model to PyTorch for scalability

## ⭐ Key Highlights

* Built neural network **from scratch (no frameworks)**
* Designed **end-to-end ML pipeline** (training → inference → API)
* Implemented **Docker-based deployment**
* Integrated **CI/CD automation with GitHub Actions**

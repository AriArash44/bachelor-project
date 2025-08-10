# 🔐 DeepSecure-IoT: AI-Powered Security System for IoT Networks

## 📘 Overview

**DeepSecure-IoT** is an intelligent cybersecurity system designed to protect IoT networks using interpretable AI models. Inspired by the research article:

> Kumar, P., Javeed, D., Islam, A. K. M. N., & Luo, X. (2025). *DeepSecure: A computational design science approach for interpretable threat hunting in cybersecurity decision making*. Decision Support Systems, 188, 114351.

This project integrates advanced machine learning, real-time threat detection, and human-readable dashboards to deliver a comprehensive security solution.

---

## 🧠 AI Architecture

- **Input Format**: IoT logs in [ToN IoT](https://research.unsw.edu.au/projects/toniot-datasets) format.
- **Modular Threat Detection**:
  - Logs are divided into **three segments**, each analyzed by a **specialist model**.
  - Each specialist uses a **MHAbiGRU** architecture to produce a **probability vector** with:
    - 1 value for **normal activity**
    - 9 values for **attack categories**
- **Ensemble Prediction**:
  - Outputs from all specialists are combined using **Logistic Regression**.
  - The highest probability in the final vector determines the **predicted label**.

---

## 🧪 Model Comparison

To validate the robustness of our architecture, we also tested and compared several alternative models

> MHAbiGRU + Logestic Regression consistently outperformed other models, especially in interpretability, performance for online threat hunting and precision across attack categories.

---

## ⚙️ Preprocessing & Analysis

- **Categorical Encoding**: DVQ-VAE (Discrete Variational Quantization - Variational Autoencoder) converts categorical features to numerical.
- **Dimensionality Reduction**: PCA + t-SNE visualizations help interpret model behavior and cluster patterns.
- **Performance Metrics**:
  - Achieved **~98%** in **accuracy**, **precision**, **recall**, and **F1-score**.

---

## 🚀 Tools

- **AI Service**:
  - **AI models**: Python + Tensorflow
  - **API Service**: Built with **FastAPI** for scalable and fast connection.
- **Web Dashboard**:
  - **Backend**: Node.js
  - **Database**: MySQL
  - **Frontend**: React + Tailwind CSS + SWR
  - Designed for **human interpretation** of AI decisions.
- **Deployment**:
  - Entire system is **containerized using Docker**.
  - **Docker Compose** orchestrates multiple services including:
    - AI Service
    - Database
    - Back-end
    - Front-end
  - Ensures seamless deployment and scalability across environments.

---

## 🎥 Demo

Watch the full system in action in this video:  
📹 _[./Demo.mp4]_  

---

## 📌 Future Work

- Extend support to additional IoT datasets (specially CSE-CIC-IDS2018).
- Increase interpretability by extracting and visualizing learned weights from the **MHABiGRU** model.
- Migrate processing from CPU to **GPU** to enhance performance during live threat hunting.
- Implement asynchronous communication using **message broker** for scalable and efficient connection handling.
- Add an **authentication service** to restrict dashboard access exclusively to authorized IoT network administrators.
- Integrate an **IoT network simulation module** to test the system under realistic threat scenarios and deployment conditions.


---

## 🧑‍💻 Author

**Arash Asghari**  
Bachelor's graduate student at **Amirkabir University of Technology (AUT)**  
📍 Karaj, Iran  
💼 Passionate about different fields in computer engineering, especially **AI** and **cybersecurity**.  
📧 arashasghari408@gmail.com

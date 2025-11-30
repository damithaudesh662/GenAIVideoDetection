# GenAIVideoDetection

# 🎥 Spotting the Unreal: Multi-Modal Deepfake Detection  
### **Final Year Project — Detecting AI-Generated Human Videos using Motion, Appearance & Geometric Cues**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)
![GPU](https://img.shields.io/badge/Compute-NVIDIA%20GPU-lightgrey.svg)

---

## 📖 Overview  
Recent video generators such as **VEO** and **Sora** can produce hyper-realistic human videos, making deepfake detection increasingly challenging.  
This project introduces a **multi-modal framework** that detects AI-generated videos based on **physical inconsistencies**, not just visual artifacts.

We combine **motion biomechanics**, **semantic appearance consistency**, and **geometric depth stability** to identify synthetic human videos.

### 🔍 Modalities Used
- **🦾 Motion** — SMPL-X fitting + RAFT optical flow  
- **🎨 Appearance** — Dense embeddings from DINOv3  
- **📐 Geometry** — Temporal depth coherence via Depth Anything V2  

## This repo has included with only using Geometric Features
---

## 🚀 Key Features

### 🧩 Multi-Modal Ensemble  
Three parallel 3D-CNN classifiers (Motion, Appearance, Geometry) fused for final prediction.

### 🌊 Geometry-Aware Detection  
Depth-based artifacts such as flickering geometry, unnatural depth boundaries, and shape inconsistencies.

### 🧠 Cross-Generator Robustness  
Generalizes across **SVD**, **Sora**, and other generative models due to reliance on physical plausibility.

---

## 🛠️ Methodology & Pipeline

### **1. Data Pipeline**
- Videos trimmed to **8 seconds**, **4 FPS**, **112×112**
- Depth maps generated using **Depth Anything V2 (ViT-S)**  
- Features extracted:
  - **Geometry:** Depth maps (Inferno colormap)

---

### **2. Classification Models**
- Backbone: **3D ResNet-18 (R3D-18)**  
- Input shape: `(B, 3, 16, 112, 112)`  
- Loss: **CrossEntropy + class balancing**  
- Training on each modality individually → ensemble fusion

---


## 💻 Installation

### **Prerequisites**
- Python **3.8+**
- CUDA-enabled GPU (recommended)

### ** Clone the Repository**
```bash
https://github.com/damithaudesh662/GenAIVideoDetection.git
cd GenAIVideoDetection


# Brain Tumour Classification App

This project is a web-based application that classifies brain tumor types from MRI scans using a deep learning model based on **DenseNet201** architecture. The app is developed using Streamlit for an interactive and user-friendly experience. It aims to assist medical professionals and researchers in early detection and diagnosis of brain tumors.

---

## Table of Contents

- [Features](#features)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)

---

## Features

- Upload and classify MRI brain scan images directly from the browser  
- Classifies tumor types such as glioma, meningioma, and pituitary tumors  
- Provides prediction confidence scores  
- Easy-to-use web interface powered by Streamlit  
- Supports image preprocessing and augmentation to improve accuracy  

---

## Dataset

The model is trained on a publicly available brain MRI dataset containing labeled images for different tumor types:

- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  
- No Tumor (Healthy)  

The dataset includes thousands of MRI images, carefully preprocessed and split into training, validation, and testing sets to ensure robust performance.

---

## Model Architecture

- Utilizes **DenseNet201**, a state-of-the-art convolutional neural network known for its dense connectivity and efficient feature reuse.  
- The DenseNet201 backbone is fine-tuned on the brain tumor MRI dataset.  
- Additional fully connected layers and dropout are applied for classification.  
- Trained with cross-entropy loss and Adam optimizer for stable convergence.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/Brain_tumourApp.git
cd Brain_tumourApp

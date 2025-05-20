# Fault Diagnosis in Analog Circuits Using Machine Learning

## Project Overview

This repository presents a comprehensive end-to-end pipeline for fault diagnosis in analog circuits, with a primary focus on the Sallen-Key Low-Pass Filter. The methodology—including data simulation, feature engineering, model development, and deployment—is generalizable to other analog circuit types for predictive maintenance and fault detection.

The highest-performing model, **XGBoost**, is deployed as an interactive Streamlit web application on Hugging Face Spaces. Users can upload test data and receive real-time fault classification results.

* [Live Web App on Hugging Face](https://huggingface.co/spaces/Shaurya-Sethi/fault-diagnosis)
* [Deployment Repository](https://github.com/Shaurya-Sethi/fault-diagnosis-app)

---

## Methodology & Approach

The project follows a structured workflow:

### 1. Data Collection

* **Circuit Simulations:**
  Data was generated using NI Multisim by simulating the Sallen-Key Low-Pass Filter under various fault scenarios, including component failures, biasing issues, and power supply faults.

### 2. Data Preprocessing & Feature Engineering

* **Preprocessing and Feature Extraction (R):**
  Raw time-domain signals were transformed into statistical features using R.
  Key features include peak-to-peak (ptp), skewness, kurtosis, zero-crossing rate (ZCR), variance, crest factor, and shape factor.
* **Interpolation and normalization** were applied to standardize the dataset.

### 3. Machine Learning & Deep Learning Models

* **XGBoost and Random Forest:**
  Both models were trained and evaluated for fault classification, with XGBoost achieving the highest accuracy. Feature importance was analyzed using SHAP values.
* **Deep Learning (PyTorch MLP):**
  A multi-layer perceptron (MLP) model was implemented in PyTorch and evaluated alongside traditional ML classifiers.

---

## Documentation

A detailed project walkthrough—including methodology, feature engineering, model training, evaluation, and insights—is available here:

* [Fault Diagnosis Report (PDF)](https://github.com/Shaurya-Sethi/fault-diagnosis-docs/blob/main/Fault%20Diagnosis%20in%20Analog%20Circuits%20Using%20Machine%20Learning%20and%20Deep%20Learning.pdf)

---

## Tools & Technologies

* **Simulation & Data Generation:** NI Multisim
* **Preprocessing & Feature Engineering:** R
* **Machine Learning & Deep Learning:** Python (scikit-learn, XGBoost, PyTorch)
* **Model Deployment:** Streamlit, Hugging Face Spaces
* **Version Control & Hosting:** GitHub

---

## Test Data for Web App

To facilitate testing, sample test data files are provided:

* [Test Files & Metadata](https://github.com/Shaurya-Sethi/fault-diagnosis-docs/tree/main/test-files)

The folder contains:

* Test circuit data files for fault detection
* `test_files_metadata.xlsx`, which documents the test data structure

You may download these files and upload them to the web app for real-time fault classification.

---

## Accessing Documentation

1. Clone the documentation repository:

   ```bash
   git clone https://github.com/Shaurya-Sethi/fault-diagnosis-docs.git
   cd fault-diagnosis-docs
   ```

2. Open `Fault_Diagnosis_Report.pdf` for a detailed project overview.

---

## Related Repositories

* **Deployment Code (Streamlit Web App):**
  [fault-diagnosis-app](https://github.com/Shaurya-Sethi/fault-diagnosis-app)
* **Full Project Documentation & Scripts:**
  [fault-diagnosis-docs](https://github.com/Shaurya-Sethi/fault-diagnosis-docs)
* **Complete Pipeline in R (Quarto, No Python):**
  [fault-diagnosis-keras-R](https://github.com/Shaurya-Sethi/fault-diagnosis-keras-R)
  *A fully reproducible pipeline for fault diagnosis in R, featuring an end-to-end Quarto notebook covering data preprocessing, feature engineering, modeling with Keras, and evaluation. No Python required.*
* **Custom Mini ML Library (Python, Autograd + MLP from Scratch):**
  [mlp-autograd-from-scratch](https://github.com/Shaurya-Sethi/mlp-autograd-from-scratch)
  *My own mini machine learning library inspired by Andrej Karpathy’s micrograd, extended to vectorized tensor operations (beyond the original scalar-only version). Features a custom autograd engine and MLP implementation from scratch. Demonstrated and validated on the fault diagnosis dataset, but applicable to any tabular data.*

**Interested in different aspects of this project?**

* [Explore the R-only workflow](https://github.com/Shaurya-Sethi/fault-diagnosis-keras-R) for rapid prototyping and full reproducibility.
* [Check out the mini ML library](https://github.com/Shaurya-Sethi/mlp-autograd-from-scratch) for an educational deep dive into building autograd and MLPs from scratch in Python.

This repository serves as the primary reference for the Fault Diagnosis in Analog Circuits project, including key code, datasets, and research documentation. For deployment-related code, see the `fault-diagnosis-app` repository.

# Fault Diagnosis in Sallen-Key Low-Pass Filters

## 📌 Project Overview
This project showcases a **complete end-to-end machine learning pipeline** for **fault diagnosis in analog circuits**. While the current implementation focuses on a **Sallen-Key Low-Pass Filter**, the methodology—including **data collection, feature engineering, model development, and deployment**—can be generalized and applied to a wide range of **analog circuits for predictive maintenance and fault detection**.

The highest-performing model, **XGBoost**, has been deployed as an interactive **Streamlit web application** on Hugging Face Spaces. The web app allows users to upload circuit test data and receive **real-time fault classification results**.

🔗 **[Live Web App on Hugging Face](https://huggingface.co/spaces/Shaurya-Sethi/fault-diagnosis)**  
🔗 **[Deployment Repository](https://github.com/Shaurya-Sethi/fault-diagnosis-app)**  

---

## **💊 Methodology & Approach**
The project follows a structured pipeline for fault diagnosis in **Sallen-Key LPFs**, leveraging simulation-based data collection and machine learning techniques. The key steps are outlined below:

### **1️⃣ Data Collection**
- **Circuit Simulations:**  
  - Data was generated using **Multisim** by simulating the **Sallen-Key Low-Pass Filter** under different fault conditions.
  - **Multiple failure modes were considered**, including open/shorted components, biasing issues, and power failures.
  
### **2️⃣ Data Preprocessing & Feature Engineering**
- **R was used for preprocessing & feature extraction**:
  - **Raw time-domain signals** were transformed into meaningful statistical features.
  - **Feature engineering pipeline** included **peak-to-peak (ptp), skewness, kurtosis, zero crossing rate (ZCR), variance, crest factor, and shape factor.**
  - **Interpolation & normalization** were performed to standardize the dataset.

### **3️⃣ Machine Learning Models**
- **XGBoost & Random Forest (RF) Classifiers**
  - Both models were trained and evaluated to detect faults.
  - **Hyperparameter tuning** was performed for **XGBoost**, which achieved the highest accuracy.
  - Feature importance analysis was conducted using **SHAP values**.

### **4️⃣ Deep Learning Models**
- **Multi-Layer Perceptron (MLP) using PyTorch**
  - A deep learning model was implemented to compare against traditional ML classifiers.
- **MLP from Scratch (Low-Level Implementation)**
  - Inspired by **Andrej Karpathy's minimalistic approach**, an MLP was built **without deep learning libraries** to demonstrate **low-level ML coding proficiency**.

---

## **📜 Documentation**
A complete walkthrough of the project, including methodology, feature engineering, model training, evaluation, and insights, is available in:  

📝 **[Fault_Diagnosis_Report.pdf](./Fault_Diagnosis_Report.pdf)** *(Detailed explanation of the entire project)*  


---

## **🛠️ Tools & Technologies Used**
- **Simulation & Data Generation**: NI Multisim  
- **Preprocessing & Feature Engineering**: R  
- **Machine Learning & Deep Learning**: Python (Scikit-learn, XGBoost, PyTorch)  
- **Model Deployment**: Streamlit, Hugging Face Spaces  
- **Version Control & Hosting**: GitHub  

---

## **🚀 How to Access the Documentation**
1. Clone this repository:  
   ```bash
   git clone https://github.com/Shaurya-Sethi/fault-diagnosis-docs.git
   cd fault-diagnosis-docs
   ```

2. Open **Fault_Diagnosis_Report.pdf** to view the complete project documentation.  

```md
🔗 Related Repositories  
1️⃣ 🖥️ Deployment Code (Streamlit Web App):  
🔗 [fault-diagnosis-app](https://github.com/Shaurya-Sethi/fault-diagnosis-app)  

2️⃣ 📝 Full Project Documentation & Scripts:  
🔗 [fault-diagnosis-docs](https://github.com/Shaurya-Sethi/fault-diagnosis-docs)  

🚀 This repository serves as a complete reference for the Fault Diagnosis project, including code, dataset, and research documentation.  
For deployment-related code, please refer to the **fault-diagnosis-app** repository.
```


# 🫀 Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-96.6%25%20Random%20Forest-2ea44f?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Heart%20Disease-blue?style=flat-square)

> An end-to-end machine learning application that predicts heart disease risk using 6 classification algorithms, deployed via Streamlit with live patient prediction.

🔗 **[Live Demo → heartsdisease-predictor.streamlit.app](https://heartsdisease-predictor.streamlit.app/)**

---

## 📊 Model Results

| Algorithm | Accuracy |
|-----------|----------|
| 🥇 Random Forest | **96.6%** |
| Logistic Regression | 90.7% |
| SVM | 90.7% |
| Naive Bayes | 86.8% |
| Decision Tree | 83.9% |
| KNN | 83.9% |

---

## 📁 Dataset

- **Source:** UCI Heart Disease Dataset
- **Records:** 1,025 patients
- **Features:** 13 clinical features (age, cholesterol, blood pressure, etc.)
- **Target:** Binary — `0` = No Disease, `1` = Disease

---

## 🖥️ App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Problem overview, dataset summary, ML pipeline |
| 📈 Dataset & EDA | Data preview, histograms, heatmap, boxplots |
| 🤖 Model Training | Algorithm comparison, confusion matrices, feature importance |
| 🔮 Predict | Live patient prediction — enter vitals, get instant result |
| 📄 Report | Full project report across all 9 phases |

---

## ⚙️ Project Phases

- [x] Phase 1: Problem Definition
- [x] Phase 2: Dataset Collection
- [x] Phase 3: Data Preprocessing
- [x] Phase 4: Exploratory Data Analysis
- [x] Phase 5: Feature Engineering
- [x] Phase 6: Model Building (6 algorithms)
- [x] Phase 7: Model Evaluation
- [x] Phase 8: Deployment (Streamlit)
- [x] Phase 9: Report

---

## 🚀 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App opens at: `http://localhost:8501`

---

## 📂 Project Structure

```
heart_disease_project/
├── app.py              ← Main Streamlit application
├── heart.csv           ← UCI Heart Disease dataset
├── models/             ← Saved trained models
├── requirements.txt    ← Python dependencies
└── README.md           ← Documentation
```

---

*ML Algorithms course project — COMSATS Institute of Information Technology, Wah Campus*

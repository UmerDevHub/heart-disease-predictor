# 🫀 Heart Disease Prediction System
### Machine Learning (CSC354) — Semester Project
**COMSATS Institute of Information Technology, Wah Campus**  
**Instructor:** Yasmeen Khaliq | **Program:** BS(SE) - 6

---

## Project Overview

An end-to-end machine learning application that predicts heart disease risk
using 6 classification algorithms from lab coursework, deployed via Streamlit.

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

The app will open at: `http://localhost:8501`

---

## Project Structure

```
heart_disease_project/
│
├── app.py              ← Main Streamlit application
├── heart.csv           ← Dataset (UCI Heart Disease)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## ML Algorithms Used (From Labs)

| Algorithm          | Lab Source | Accuracy |
|--------------------|------------|----------|
| Logistic Regression| Lab 4      | ~90.7%   |
| Decision Tree      | Lab 5      | ~83.9%   |
| Random Forest      | Lab 5      | ~96.6%   |
| SVM                | Lab 8      | ~90.7%   |
| KNN                | Lab 7      | ~83.9%   |
| Naive Bayes        | Lab 6      | ~86.8%   |

---

## App Pages

1. **Home** — Problem overview, dataset summary, ML pipeline steps
2. **Dataset & EDA** — Data preview, histograms, heatmap, boxplots
3. **Model Training** — Algorithm comparison, confusion matrices, feature importance
4. **Predict** — Live patient prediction with any model
5. **Report** — Full project report (all 9 phases)

---

## Project Phases Covered

- [x] Phase 1: Problem Definition
- [x] Phase 2: Dataset Collection
- [x] Phase 3: Data Preprocessing
- [x] Phase 4: EDA
- [x] Phase 5: Feature Engineering
- [x] Phase 6: Model Building (6 algorithms)
- [x] Phase 7: Model Evaluation
- [x] Phase 8: Deployment (Streamlit)
- [x] Phase 9: Report

---

## Dataset

- **Source:** UCI Heart Disease Dataset
- **Records:** 1025
- **Features:** 13 clinical features
- **Target:** 0 = No Disease, 1 = Disease

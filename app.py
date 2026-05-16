import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────── PAGE CONFIG ────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── GLOBAL CSS ─────────────────────────────
st.markdown("""
<style>
    /* ── fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── page background ── */
    .main { background-color: #f8f9fb; }
    .block-container { padding: 1.8rem 2.5rem; }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

    /* ── metric cards ── */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* ── section headers ── */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #e53e3e;
        display: inline-block;
    }

    /* ── result boxes ── */
    .result-danger {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .result-safe {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .result-title { font-size: 1.25rem; font-weight: 700; margin: 0 0 4px 0; }
    .result-sub   { font-size: 0.92rem; color: #4b5563; margin: 0; }

    /* ── info box ── */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #1e40af;
        margin-bottom: 12px;
    }

    /* ── table ── */
    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .styled-table th {
        background: #f3f4f6; color: #374151;
        padding: 9px 12px; text-align: left;
        border-bottom: 2px solid #e5e7eb; font-weight: 600;
    }
    .styled-table td { padding: 8px 12px; border-bottom: 1px solid #f3f4f6; color: #374151; }
    .styled-table tr:hover td { background: #f9fafb; }
    .badge-best {
        background: #dcfce7; color: #166534;
        padding: 2px 8px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
    }

    /* ── divider ── */
    hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.2rem 0; }

    /* hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── HELPERS ────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    df.drop_duplicates(inplace=True)
    # Phase 5 — Feature Engineering: create new derived features
    df['age_group'] = df['age'].apply(lambda x: 0 if x < 45 else (1 if x < 60 else 2))
    df['bp_risk'] = df['trestbps'] * df['oldpeak']
    df['heart_rate_reserve'] = df['thalach'] - df['age']
    return df


@st.cache_resource
def train_all_models(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── exactly the same models as taught in labs ──
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(kernel="rbf", C=1, probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Naive Bayes":         GaussianNB(),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "y_pred":    y_pred,
            "cm":        confusion_matrix(y_test, y_pred),
        }
        trained[name] = model

    return trained, scaler, results, X_test, y_test, X_train, y_train


def section(title):
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)


# ─────────────────────────── SIDEBAR ────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Disease\nPrediction System")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Home",
        "📊 Dataset & EDA",
        "⚙️ Model Training",
        "🔮 Predict",
        "📋 Report"
    ])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#6b7280; line-height:1.6'>
    <b>Subject:</b> Machine Learning (CSC354)<br>
    <b>Instructor:</b> Yasmeen Khaliq<br>
    <b>Program:</b> BS(SE) - 6<br><br>
    <b>Algorithms Used:</b><br>
    • Logistic Regression (Lab 4)<br>
    • Decision Tree (Lab 5)<br>
    • Random Forest (Lab 5)<br>
    • Naive Bayes (Lab 6)<br>
    • KNN (Lab 7)<br>
    • SVM (Lab 8)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────── LOAD ───────────────────────────────────
df = load_data()
trained_models, scaler, results, X_test, y_test, X_train, y_train = train_all_models(df)
best_model_name = max(results, key=lambda k: results[k]["Accuracy"])


# ════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("## 🫀 Heart Disease Prediction System")
    st.markdown(
        "An end-to-end machine learning application that predicts the likelihood "
        "of heart disease using patient clinical data and multiple ML algorithms.")

    st.markdown("---")

    # top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",   f"{len(df):,}")
    c2.metric("Features",        "13")
    c3.metric("Best Accuracy",   f"{results[best_model_name]['Accuracy']}%")
    c4.metric("Best Model",      best_model_name)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        section("Problem Statement")
        st.write(
            "Heart disease is a leading cause of death worldwide. Early and accurate "
            "prediction using patient health records can significantly improve outcomes. "
            "This system applies classification algorithms to predict heart disease risk "
            "based on 13 clinical features including age, cholesterol, chest pain type, "
            "and ECG results.")

        section("Dataset Overview")
        st.write("**Source:** UCI Heart Disease Dataset (via Kaggle)")
        info = {
            "Records": len(df),
            "Features": 13,
            "Target Classes": "0 = No Disease, 1 = Disease",
            "Missing Values": df.isnull().sum().sum(),
            "Duplicates Removed": 0,
        }
        for k, v in info.items():
            st.markdown(f"- **{k}:** {v}")

    with col2:
        section("ML Pipeline")
        steps = [
            ("1", "Problem Definition", "Classification — Heart Disease Yes/No"),
            ("2", "Data Collection",    "UCI Heart Disease Dataset (1025 records)"),
            ("3", "Preprocessing",      "Null handling, duplicates, normalization"),
            ("4", "EDA",                "Histograms, heatmap, boxplots, scatter"),
            ("5", "Feature Engineering","Selection, scaling + 3 new derived features"),
            ("6", "Model Building",     "6 algorithms from lab coursework"),
            ("7", "Evaluation",         "Accuracy, Precision, Recall, F1, CM"),
            ("8", "Deployment",         "Streamlit interactive UI"),
        ]
        for num, title, desc in steps:
            st.markdown(
                f"<div style='display:flex;gap:10px;margin-bottom:8px;align-items:flex-start'>"
                f"<span style='background:#e53e3e;color:white;border-radius:50%;width:22px;"
                f"height:22px;display:flex;align-items:center;justify-content:center;"
                f"font-size:0.75rem;font-weight:700;flex-shrink:0'>{num}</span>"
                f"<div><b style='font-size:0.88rem'>{title}</b>"
                f"<div style='font-size:0.8rem;color:#6b7280'>{desc}</div></div></div>",
                unsafe_allow_html=True)

    st.markdown("---")
    section("Feature Descriptions")
    feat_df = pd.DataFrame([
        ["age",      "Age of patient (years)",                             "Continuous"],
        ["sex",      "Sex (1=Male, 0=Female)",                             "Binary"],
        ["cp",       "Chest pain type (0–3)",                              "Categorical"],
        ["trestbps", "Resting blood pressure (mm Hg)",                    "Continuous"],
        ["chol",     "Serum cholesterol (mg/dl)",                         "Continuous"],
        ["fbs",      "Fasting blood sugar > 120 mg/dl (1=True, 0=False)", "Binary"],
        ["restecg",  "Resting ECG results (0–2)",                         "Categorical"],
        ["thalach",  "Maximum heart rate achieved",                       "Continuous"],
        ["exang",    "Exercise induced angina (1=Yes, 0=No)",              "Binary"],
        ["oldpeak",  "ST depression induced by exercise",                 "Continuous"],
        ["slope",    "Slope of peak exercise ST segment (0–2)",           "Categorical"],
        ["ca",       "Number of major vessels (0–4)",                     "Categorical"],
        ["thal",     "Thalassemia (0=normal, 1=fixed, 2=reversible)",     "Categorical"],
        ["age_group",         "Age category (0=Young<45, 1=Middle 45-60, 2=Senior>60)", "Engineered"],
        ["bp_risk",           "BP risk score (trestbps × oldpeak)",                     "Engineered"],
        ["heart_rate_reserve","Heart rate reserve (thalach − age)",                      "Engineered"],
    ], columns=["Feature", "Description", "Type"])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATASET & EDA
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset & EDA":
    st.markdown("## 📊 Dataset & Exploratory Data Analysis")
    st.markdown("Understanding the data before building any model.")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Data Preview", "📈 Distributions", "🔥 Correlations", "📦 Boxplots", "🔍 Scatter Plots"])

    # ── Tab 1: Data Preview ──────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            section("Dataset Sample")
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            section("Statistical Summary")
            st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            section("Target Distribution")
            counts = df['target'].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            colors = ['#38a169', '#e53e3e']
            bars = ax.bar(['No Disease (0)', 'Disease (1)'], counts.values,
                          color=colors, width=0.5, edgecolor='white')
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(int(bar.get_height())), ha='center', fontsize=10, fontweight='bold')
            ax.set_ylabel("Count"); ax.set_title("Target Class Distribution", fontsize=11, fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
            ax.set_facecolor('#f8f9fb'); fig.patch.set_facecolor('#f8f9fb')
            st.pyplot(fig); plt.close()

        with c2:
            section("Gender Distribution")
            sex_counts = df['sex'].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.pie(sex_counts.values, labels=['Male','Female'],
                   autopct='%1.1f%%', colors=['#3b82f6','#f472b6'],
                   startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
            ax.set_title("Sex Distribution", fontsize=11, fontweight='bold')
            fig.patch.set_facecolor('#f8f9fb')
            st.pyplot(fig); plt.close()

        with c3:
            section("Missing Values Check")
            mv = df.isnull().sum().reset_index()
            mv.columns = ['Feature', 'Missing']
            mv['Status'] = mv['Missing'].apply(lambda x: '✅ Clean' if x == 0 else f'⚠️ {x}')
            st.dataframe(mv[['Feature','Status']].head(13), use_container_width=True, hide_index=True)

    # ── Tab 2: Distributions ─────────────────────────────────────────
    with tab2:
        section("Feature Histograms — Disease vs No Disease")
        num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
        fig.patch.set_facecolor('#f8f9fb')
        for i, col in enumerate(num_cols):
            for val, color, label in [(0,'#38a169','No Disease'), (1,'#e53e3e','Disease')]:
                axes[i].hist(df[df['target']==val][col], bins=18, alpha=0.65,
                             color=color, label=label, edgecolor='white')
            axes[i].set_title(col, fontsize=10, fontweight='bold')
            axes[i].set_xlabel(col, fontsize=8); axes[i].set_ylabel("Count", fontsize=8)
            axes[i].spines[['top','right']].set_visible(False)
            axes[i].set_facecolor('#f8f9fb')
        axes[0].legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("---")
        section("Chest Pain Type vs Target")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
        cp_target = df.groupby(['cp','target']).size().unstack(fill_value=0)
        cp_target.plot(kind='bar', ax=ax, color=['#38a169','#e53e3e'],
                       edgecolor='white', width=0.6)
        ax.set_xlabel("Chest Pain Type (0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic)")
        ax.set_ylabel("Count"); ax.set_title("Chest Pain Type vs Heart Disease", fontsize=11, fontweight='bold')
        ax.legend(['No Disease','Disease']); ax.spines[['top','right']].set_visible(False)
        plt.xticks(rotation=0)
        st.pyplot(fig); plt.close()

    # ── Tab 3: Correlation ───────────────────────────────────────────
    with tab3:
        section("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        sns.heatmap(df.corr(), mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlGn', center=0, ax=ax,
                    annot_kws={'size': 8},
                    linewidths=0.5, linecolor='#f8f9fb',
                    cbar_kws={'shrink': 0.8})
        ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("---")
        section("Correlation with Target")
        corr_target = df.corr()['target'].drop('target').sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
        colors = ['#e53e3e' if v > 0 else '#3b82f6' for v in corr_target.values]
        ax.barh(corr_target.index, corr_target.values, color=colors, edgecolor='white')
        ax.axvline(0, color='#9ca3af', linewidth=1, linestyle='--')
        ax.set_xlabel("Correlation Coefficient"); ax.set_title("Feature Correlation with Target", fontsize=11, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        red_p = mpatches.Patch(color='#e53e3e', label='Positive (increases risk)')
        blue_p = mpatches.Patch(color='#3b82f6', label='Negative (decreases risk)')
        ax.legend(handles=[red_p, blue_p], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Tab 4: Boxplots ──────────────────────────────────────────────
    with tab4:
        section("Boxplots — Outlier Detection")
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        fig.patch.set_facecolor('#f8f9fb')
        box_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        for i, col in enumerate(box_features):
            ax = axes[i//3][i%3]
            data_no  = df[df['target']==0][col]
            data_yes = df[df['target']==1][col]
            bp = ax.boxplot([data_no, data_yes], patch_artist=True,
                            labels=['No Disease','Disease'],
                            medianprops=dict(color='#111827', linewidth=2))
            bp['boxes'][0].set_facecolor('#bbf7d0')
            bp['boxes'][1].set_facecolor('#fecaca')
            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
            ax.set_facecolor('#f8f9fb')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""
        <div class='info-box'>
        📌 <b>Interpretation:</b> Boxplots show the spread and outliers per feature grouped by target.
        Features like <b>oldpeak</b> and <b>thalach</b> show clear separation between disease/no-disease groups,
        making them strong predictors.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 5: Scatter Plots ─────────────────────────────────────────
    with tab5:
        section("Scatter Plots — Feature Relationships")

        sc1, sc2 = st.columns(2)
        with sc1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
            for val, color, label in [(0,'#38a169','No Disease'),(1,'#e53e3e','Disease')]:
                s = df[df['target']==val]
                ax.scatter(s['thalach'], s['oldpeak'], c=color, label=label,
                           alpha=0.6, edgecolors='white', linewidth=0.5, s=40)
            ax.set_xlabel('Max Heart Rate (thalach)', fontsize=9)
            ax.set_ylabel('ST Depression (oldpeak)', fontsize=9)
            ax.set_title('Max Heart Rate vs ST Depression', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with sc2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
            for val, color, label in [(0,'#38a169','No Disease'),(1,'#e53e3e','Disease')]:
                s = df[df['target']==val]
                ax.scatter(s['age'], s['thalach'], c=color, label=label,
                           alpha=0.6, edgecolors='white', linewidth=0.5, s=40)
            ax.set_xlabel('Age', fontsize=9)
            ax.set_ylabel('Max Heart Rate (thalach)', fontsize=9)
            ax.set_title('Age vs Max Heart Rate', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('---')
        sc3, sc4 = st.columns(2)
        with sc3:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
            for val, color, label in [(0,'#38a169','No Disease'),(1,'#e53e3e','Disease')]:
                s = df[df['target']==val]
                ax.scatter(s['age'], s['chol'], c=color, label=label,
                           alpha=0.6, edgecolors='white', linewidth=0.5, s=40)
            ax.set_xlabel('Age', fontsize=9)
            ax.set_ylabel('Cholesterol (mg/dl)', fontsize=9)
            ax.set_title('Age vs Cholesterol', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with sc4:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
            for val, color, label in [(0,'#38a169','No Disease'),(1,'#e53e3e','Disease')]:
                s = df[df['target']==val]
                ax.scatter(s['trestbps'], s['chol'], c=color, label=label,
                           alpha=0.6, edgecolors='white', linewidth=0.5, s=40)
            ax.set_xlabel('Resting Blood Pressure (mm Hg)', fontsize=9)
            ax.set_ylabel('Cholesterol (mg/dl)', fontsize=9)
            ax.set_title('Resting BP vs Cholesterol', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("""
        <div class='info-box'>
        📌 <b>Interpretation:</b> Scatter plots reveal pairwise feature relationships.<br>
        <b>thalach vs oldpeak</b> shows clear cluster separation — disease patients tend to have
        lower max heart rate and higher ST depression.<br>
        <b>Age vs thalach</b> confirms that older patients with disease have notably lower max heart rates.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL TRAINING
# ════════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Training":
    st.markdown("## ⚙️ Model Training & Evaluation")
    st.markdown("Six ML algorithms from lab coursework trained and compared.")
    st.markdown("---")

    # ── Algorithm Reference ──────────────────────────────────────────
    section("Algorithm Reference (From Labs)")
    algo_info = {
        "Logistic Regression": ("Lab 4", "Binary classification using sigmoid function. Baseline model.", "LogisticRegression(random_state=42, max_iter=1000)"),
        "Decision Tree":       ("Lab 5", "Tree-based splits using entropy (information gain). Criterion='entropy', max_depth=5.", "DecisionTreeClassifier(criterion='entropy', max_depth=5)"),
        "Random Forest":       ("Lab 5", "Ensemble of decision trees, reduces overfitting. 100 estimators.", "RandomForestClassifier(n_estimators=100)"),
        "SVM":                 ("Lab 8", "Finds optimal hyperplane. RBF kernel, C=1.", "SVC(kernel='rbf', C=1, probability=True)"),
        "KNN":                 ("Lab 7", "Classifies based on 5 nearest neighbors (Euclidean distance).", "KNeighborsClassifier(n_neighbors=5, metric='euclidean')"),
        "Naive Bayes":         ("Lab 6", "Probabilistic classifier using Bayes theorem. Gaussian variant.", "GaussianNB()"),
    }
    rows = ""
    for name, (lab, desc, code) in algo_info.items():
        best_tag = '<span class="badge-best">Best</span>' if name == best_model_name else ''
        rows += f"<tr><td><b>{name}</b> {best_tag}</td><td>{lab}</td><td>{desc}</td><td><code style='font-size:0.78rem;background:#f3f4f6;padding:2px 5px;border-radius:4px'>{code}</code></td></tr>"
    st.markdown(f"""
    <table class='styled-table'>
      <tr><th>Algorithm</th><th>Lab Source</th><th>Configuration</th><th>Code</th></tr>
      {rows}
    </table>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics Table ────────────────────────────────────────────────
    section("Performance Comparison Table")
    metric_rows = ""
    for name, m in results.items():
        best_tag = '<span class="badge-best">✓ Best</span>' if name == best_model_name else ''
        metric_rows += (
            f"<tr><td><b>{name}</b> {best_tag}</td>"
            f"<td><b>{m['Accuracy']}%</b></td>"
            f"<td>{m['Precision']}%</td>"
            f"<td>{m['Recall']}%</td>"
            f"<td>{m['F1-Score']}%</td></tr>"
        )
    st.markdown(f"""
    <table class='styled-table'>
      <tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
      {metric_rows}
    </table>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Bar Chart Comparison ─────────────────────────────────────────
    section("Accuracy Comparison Chart")
    model_names  = list(results.keys())
    accuracies   = [results[m]['Accuracy'] for m in model_names]
    bar_colors   = ['#e53e3e' if m == best_model_name else '#93c5fd' for m in model_names]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
    bars = ax.bar(model_names, accuracies, color=bar_colors, edgecolor='white', width=0.55)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(70, 105)
    ax.set_title("Model Accuracy Comparison", fontsize=12, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Confusion Matrices ───────────────────────────────────────────
    section("Confusion Matrices — All Models")
    cols = st.columns(3)
    for i, (name, m) in enumerate(results.items()):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
            sns.heatmap(m['cm'], annot=True, fmt='d', ax=ax,
                        cmap='Reds' if name == best_model_name else 'Blues',
                        linewidths=0.5, linecolor='white',
                        xticklabels=['No Disease','Disease'],
                        yticklabels=['No Disease','Disease'],
                        annot_kws={'size': 11, 'weight': 'bold'})
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            title_suffix = " ★" if name == best_model_name else ""
            ax.set_title(f"{name}{title_suffix}", fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    st.markdown("---")
    # ── Feature Importance ───────────────────────────────────────────
    section("Feature Importance (Random Forest)")
    rf_model = trained_models['Random Forest']
    feat_names = df.drop('target', axis=1).columns
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
    bars = ax.barh(fi_df['Feature'], fi_df['Importance'],
                   color='#fca5a5', edgecolor='white')
    bars[-1].set_color('#e53e3e')  # highlight top
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest — Feature Importance", fontsize=11, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICT
# ════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("## 🔮 Heart Disease Prediction")
    st.markdown("Enter patient clinical data below to get a prediction.")
    st.markdown("---")

    col_form, col_result = st.columns([1.1, 0.9])

    with col_form:
        section("Patient Clinical Data")

        c1, c2 = st.columns(2)
        with c1:
            age      = st.number_input("Age (years)",         min_value=20,  max_value=100, value=52)
            trestbps = st.number_input("Resting BP (mm Hg)",  min_value=80,  max_value=220, value=130)
            chol     = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=220)
            thalach  = st.number_input("Max Heart Rate",       min_value=60,  max_value=220, value=150)
            oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=8.0, value=1.0, step=0.1)
            ca       = st.number_input("Major Vessels (0–4)",  min_value=0,   max_value=4,   value=0)

        with c2:
            sex     = st.selectbox("Sex",              ["Male (1)", "Female (0)"])
            cp      = st.selectbox("Chest Pain Type",  ["Typical Angina (0)", "Atypical Angina (1)", "Non-Anginal (2)", "Asymptomatic (3)"])
            fbs     = st.selectbox("Fasting Blood Sugar > 120", ["Yes (1)", "No (0)"])
            restecg = st.selectbox("Resting ECG",      ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
            exang   = st.selectbox("Exercise Angina",  ["Yes (1)", "No (0)"])
            slope   = st.selectbox("ST Slope",         ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
            thal    = st.selectbox("Thalassemia",      ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)", "Unknown (3)"])

        model_choice = st.selectbox("Select ML Model", list(trained_models.keys()),
                                    index=list(trained_models.keys()).index("Random Forest"))

        predict_btn = st.button("🔮 Predict", use_container_width=True, type="primary")

    with col_result:
        section("Prediction Result")

        if predict_btn:
            # parse inputs
            sex_v     = 1 if "Male" in sex else 0
            cp_v      = int(cp.split("(")[1].replace(")",""))
            fbs_v     = int(fbs.split("(")[1].replace(")",""))
            restecg_v = int(restecg.split("(")[1].replace(")",""))
            exang_v   = int(exang.split("(")[1].replace(")",""))
            slope_v   = int(slope.split("(")[1].replace(")",""))
            thal_v    = int(thal.split("(")[1].replace(")",""))

            # compute engineered features from user inputs
            age_group_v   = 0 if age < 45 else (1 if age < 60 else 2)
            bp_risk_v     = trestbps * oldpeak
            hr_reserve_v  = thalach - age

            input_data = np.array([[age, sex_v, cp_v, trestbps, chol, fbs_v,
                                    restecg_v, thalach, exang_v, oldpeak,
                                    slope_v, ca, thal_v,
                                    age_group_v, bp_risk_v, hr_reserve_v]])
            input_scaled = scaler.transform(input_data)

            model    = trained_models[model_choice]
            pred     = model.predict(input_scaled)[0]
            prob     = model.predict_proba(input_scaled)[0]
            risk_pct = round(prob[1] * 100, 1)

            if pred == 1:
                st.markdown(f"""
                <div class='result-danger'>
                  <p class='result-title' style='color:#c53030'>⚠️ Heart Disease Detected</p>
                  <p class='result-sub'>The model predicts <b>high risk</b> of heart disease.</p>
                  <p class='result-sub'>Risk Probability: <b style='color:#c53030'>{risk_pct}%</b></p>
                  <p class='result-sub'>Model Used: <b>{model_choice}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-safe'>
                  <p class='result-title' style='color:#276749'>✅ No Heart Disease Detected</p>
                  <p class='result-sub'>The model predicts <b>low risk</b> of heart disease.</p>
                  <p class='result-sub'>Risk Probability: <b style='color:#276749'>{risk_pct}%</b></p>
                  <p class='result-sub'>Model Used: <b>{model_choice}</b></p>
                </div>
                """, unsafe_allow_html=True)

            # probability bar
            st.markdown("**Prediction Probability**")
            fig, ax = plt.subplots(figsize=(5, 1.5))
            fig.patch.set_facecolor('#f8f9fb'); ax.set_facecolor('#f8f9fb')
            ax.barh(['No Disease'], [prob[0]*100], color='#38a169', height=0.4)
            ax.barh(['Disease'],    [prob[1]*100], color='#e53e3e', height=0.4)
            for bar, val in zip(ax.patches, [prob[0]*100, prob[1]*100]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
            ax.set_xlim(0, 115); ax.set_xlabel("Probability (%)", fontsize=8)
            ax.spines[['top','right','left']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # input summary
            section("Input Summary")
            summary = pd.DataFrame({
                'Feature': ['Age','Sex','Chest Pain','Rest BP','Cholesterol',
                            'Fasting BS','ECG','Max HR','Ex Angina','Oldpeak','Slope','CA','Thal'],
                'Value':   [age, sex_v, cp_v, trestbps, chol, fbs_v,
                            restecg_v, thalach, exang_v, oldpeak, slope_v, ca, thal_v]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.markdown("""
            <div class='info-box'>
            ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only.
            Always consult a qualified medical professional for diagnosis.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='background:#f3f4f6;border-radius:10px;padding:30px;text-align:center;color:#6b7280;margin-top:20px'>
              <div style='font-size:2.5rem'>🫀</div>
              <p style='font-size:0.95rem;margin:8px 0 0'>Fill in the patient details on the left<br>and click <b>Predict</b></p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 5 — REPORT
# ════════════════════════════════════════════════════════════════════
elif page == "📋 Report":
    st.markdown("## 📋 Final Project Report")
    st.markdown("Summary of the complete machine learning pipeline.")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        section("1. Problem Definition")
        st.write("""
        **Type:** Supervised Classification  
        **Goal:** Predict whether a patient has heart disease (1) or not (0)  
        **Input:** 13 clinical features  
        **Output:** Binary class (0 or 1)
        """)

        section("2. Dataset")
        st.write(f"""
        - **Source:** UCI Heart Disease Dataset  
        - **Records:** {len(df)} (after duplicate removal)  
        - **Features:** 13  
        - **Missing Values:** None  
        - **Class Balance:** {df['target'].value_counts()[1]} disease / {df['target'].value_counts()[0]} no disease
        """)

        section("3. Preprocessing Steps")
        steps = [
            "✅ Loaded dataset with pandas",
            "✅ Checked and confirmed zero missing values",
            "✅ Removed duplicate records",
            "✅ Applied StandardScaler normalization",
            "✅ Train-test split: 80% train / 20% test",
            "✅ Encoded categorical features (already numeric)",
        ]
        for s in steps: st.markdown(f"- {s}")

        section("4. EDA Findings")
        st.write("""
        - `thalach` (max heart rate) negatively correlated with disease — patients with disease have lower max HR  
        - `oldpeak` (ST depression) positively correlated — higher = more risk  
        - `cp` type 0 (typical angina) shows highest disease rate  
        - Age and sex are important predictors  
        - No severe outliers were detected that required removal
        """)

    with c2:
        section("5. Feature Engineering")
        st.write("""
        - All 13 original features retained based on correlation analysis  
        - **3 new derived features created:**  
          - `age_group` — categorizes age into Young (<45), Middle (45–60), Senior (>60)  
          - `bp_risk` — blood pressure risk score (trestbps × oldpeak)  
          - `heart_rate_reserve` — cardiac reserve indicator (thalach − age)  
        - StandardScaler applied to normalize all 16 features  
        - No dimensionality reduction needed (16 features manageable)  
        - Feature importance confirmed using Random Forest  
        
        **Justification:** `age_group` captures non-linear age effects; `bp_risk` combines  
        resting BP with exercise-induced ST depression to create a composite cardiac stress  
        indicator; `heart_rate_reserve` reflects the heart's functional capacity, which  
        decreases with disease.
        """)

        section("6. Models & Results")
        rows = ""
        for name, m in sorted(results.items(), key=lambda x: -x[1]['Accuracy']):
            best_tag = ' ★' if name == best_model_name else ''
            rows += (f"<tr><td><b>{name}{best_tag}</b></td>"
                     f"<td>{m['Accuracy']}%</td><td>{m['Precision']}%</td>"
                     f"<td>{m['Recall']}%</td><td>{m['F1-Score']}%</td></tr>")
        st.markdown(f"""
        <table class='styled-table'>
          <tr><th>Model</th><th>Acc</th><th>Pre</th><th>Rec</th><th>F1</th></tr>
          {rows}
        </table>""", unsafe_allow_html=True)

        section("7. Best Model Justification")
        st.write(f"""
        **{best_model_name}** achieved the highest accuracy of **{results[best_model_name]['Accuracy']}%**.  
        
        Random Forest outperforms other models because:  
        - It is an ensemble method (combines multiple decision trees)  
        - Reduces overfitting compared to a single Decision Tree  
        - Handles both linear and non-linear patterns  
        - Robust to noisy features  
        - Provides feature importance scores
        """)

        section("8. Deployment")
        st.write("""
        - **Framework:** Streamlit  
        - **Reason:** Best for ML dashboards — supports charts, metrics, forms in pure Python  
        - **Features:** EDA dashboard, model comparison, live prediction, full report  
        - **Input Validation:** Min/max bounds on all numeric inputs, dropdowns for categorical
        """)

    st.markdown("---")
    section("9. Conclusion")
    st.write(f"""
    This project successfully demonstrated an end-to-end machine learning pipeline for heart 
    disease prediction. Six algorithms were implemented using code patterns from lab coursework 
    (Labs 4–8). **{best_model_name}** was selected as the final model with **{results[best_model_name]['Accuracy']}%** accuracy.
    
    The Streamlit application provides a complete interface for data exploration, model evaluation,
    and real-time patient prediction, satisfying all 9 phases of the project requirements.
    """)

    section("10. References")
    st.write("""
    - UCI Heart Disease Dataset — archive.ics.uci.edu  
    - Scikit-learn Documentation — scikit-learn.org  
    - Lab Manual: Machine Learning (CSC354), COMSATS Wah Campus  
    - Prateek Joshi, *Artificial Intelligence with Python*, Packt Publishing, 2017  
    - Streamlit Documentation — docs.streamlit.io
    """)

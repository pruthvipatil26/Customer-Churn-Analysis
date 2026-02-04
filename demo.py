import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, auc,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG & ENHANCED PREMIUM SAAS STYLING WITH ANIMATIONS
# =====================================================
st.set_page_config(page_title="Customer Churn Analytics ", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Global Theme */
.main {
    background-color: #F9FAFC;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Typography */
h1, h2, h3 {
    color: #1F2937;
    font-weight: 700;
    letter-spacing: -0.025em;
}
h1 { font-size: 2.5rem; }
h2 { font-size: 1.75rem; }
h3 { font-size: 1.35rem; }

/* Hero Header with Fade-In Animation */
.header-banner {
    background: linear-gradient(135deg, #6366F1, #06B6D4);
    padding: 3.5rem;
    border-radius: 24px;
    color: white;
    text-align: center;
    box-shadow: 0 12px 50px rgba(99, 102, 241, 0.25);
    animation: fadeInDown 1s ease-out;
}

/* Glassmorphism Cards with Enhanced Hover & Entrance Animation */
[data-testid="metric-container"], .stAlert, .stExpander, .prediction-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 18px;
    box-shadow: 0 6px 30px rgba(0, 0, 0, 0.06);
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    opacity: 0;
    animation: fadeInUp 0.8s ease-out forwards;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="metric-container"]:hover, .stAlert:hover, .stExpander:hover, .prediction-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 50px rgba(99, 102, 241, 0.15);
}

/* Staggered Animation Delay for Cards */
[data-testid="metric-container"]:nth-child(1) { animation-delay: 0.2s; }
[data-testid="metric-container"]:nth-child(2) { animation-delay: 0.4s; }
[data-testid="metric-container"]:nth-child(3) { animation-delay: 0.6s; }

/* Button with Pulse & Lift Animation */
.stButton > button {
    background: linear-gradient(90deg, #6366F1, #06B6D4);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 600;
    padding: 0.85rem 2.5rem;
    font-size: 1.1rem;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
    transition: all 0.4s ease;
    animation: fadeInUp 0.8s ease-out 0.8s both;
}
.stButton > button:hover {
    transform: translateY(-4px) scale(1.05);
    box-shadow: 0 12px 30px rgba(99, 102, 241, 0.4);
}
.stButton > button:active {
    transform: translateY(0) scale(0.98);
}

/* Sidebar with Smooth Highlight & Icon Animation */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6366F1, #3B82F6);
    padding: 2rem 1rem;
    border-radius: 0 28px 28px 0;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: white !important;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 0.5rem;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}
section[data-testid="stSidebar"] .stRadio > label::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.6s;
}
section[data-testid="stSidebar"] .stRadio > label:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateX(12px);
    padding-left: 1.8rem;
}
section[data-testid="stSidebar"] .stRadio > label:hover::before {
    left: 100%;
}
section[data-testid="stSidebar"] .stRadio > input:checked + label {
    background: rgba(255, 255, 255, 0.25);
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* Section Divider with Fade */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(to right, transparent, #C7D2FE, transparent);
    margin: 3rem 0;
    animation: fadeIn 1.2s ease-out;
}

/* Plot Containers with Subtle Entrance */
.stPyplot {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
    animation: fadeInUp 0.9s ease-out;
    background: white;
    padding: 1rem;
}

/* Input Fields with Focus Glow */
.stSelectbox > div > div, .stSlider > div, .stNumberInput > div {
    border-radius: 14px;
    border: 1px solid #E0E7FF;
    background: white;
    transition: all 0.3s ease;
}
.stSelectbox > div > div:focus-within, .stSlider > div:hover, .stNumberInput > div:focus-within {
    border-color: #6366F1;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
    transform: scale(1.02);
}

/* Prediction Badge with Scale & Glow Animation */
.prediction-card {
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    animation: scaleIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}
.low-risk { 
    background: linear-gradient(135deg, #10B981, #34D399); 
    color: white; 
}
.high-risk { 
    background: linear-gradient(135deg, #EF4444, #F87171); 
    color: white; 
}

/* Progress Bar Animation */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366F1, #06B6D4);
    animation: progressFill 1.5s ease-out;
}

/* Keyframes for Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-40px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes scaleIn {
    0% { opacity: 0; transform: scale(0.8); }
    70% { transform: scale(1.08); }
    100% { opacity: 1; transform: scale(1); }
}
@keyframes progressFill {
    from { width: 0%; }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA (CACHED)
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/churn_dataset (1).csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# =====================================================
# PREPARE MODELS (CACHED)
# =====================================================
@st.cache_resource
def prepare_models():
    df_ml = pd.get_dummies(df.drop('customerID', axis=1, errors='ignore'), drop_first=True)
    X = df_ml.drop("Churn", axis=1)
    y = df_ml["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = lr
    
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_scaled, y_train)
    models["Decision Tree"] = dt
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    models["Random Forest"] = rf
    
    return models, scaler, X.columns.tolist(), X_test_scaled, y_test

models_dict, scaler, feature_columns, X_test_scaled, y_test = prepare_models()

# =====================================================
# HERO HEADER (Animated)
# =====================================================
st.markdown("""
<div class="header-banner">
<h1>Customer Churn Analytics </h1>
<p style="font-size:1.35rem; opacity:0.95; margin-top:0.8rem;">
 EDA, Real-Time ML Prediction & Actionable Insights
</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Cards (Staggered Fade-In)
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Customers", f"{df.shape[0]:,}")
col2.metric("📉 Churn Rate", f"{df['Churn'].mean()*100:.2f}%")
col3.metric("📊 Features Analyzed", df.shape[1])

st.markdown("<hr>", unsafe_allow_html=True)

# =====================================================
# ENHANCED SIDEBAR NAVIGATION (Clean Footer)
# =====================================================
st.sidebar.markdown("<h3 style='color:white; text-align:center; margin-top:-10px;'>Customer Churn Analysis </h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Navigate",
    options=["eda", "ml", "insights"],
    format_func=lambda x: {
        "eda": "📊 Data Exploration",
        "ml": "🤖 ML Prediction ",
        "insights": "💡 Strategic Insights"
    }[x]
)

# REMOVED the dynamic description line that said "Explore patterns and distributions interactively"

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Pruthviraj Patil")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'customerID']

# =====================================================
# DYNAMIC PAGE TITLE
# =====================================================
page_titles = {
    "eda": "📊 Exploratory Data Analysis",
    "ml": "🤖 Machine Learning Prediction Engine",
    "insights": "💡 Strategic Business Insights"
}
st.markdown(f"## {page_titles[section]}")

# =====================================================
# EDA SECTION (FULLY RESTORED INCLUDING BIVARIATE)
# =====================================================
if section == "eda":
    # REMOVED the line: st.markdown("Interactive visualizations to uncover customer behavior patterns.")
    
    with st.expander("🔍 Configure Analysis", expanded=True):
        analysis_type = st.radio("Analysis Mode", ["Univariate", "Bivariate"], horizontal=True)
    
    if analysis_type == "Univariate":
        col = st.selectbox("Select Feature", num_cols + cat_cols)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 📈 Distribution Overview")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.set_style("whitegrid")
            sns.set_palette("viridis")
            plt.rcParams.update({'font.size': 12})
            if col in num_cols:
                sns.histplot(df[col], kde=True, ax=ax)
            else:
                sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
                plt.xticks(rotation=45)
            plt.title(f"{col} Distribution")
            st.pyplot(fig)
        
        with c2:
            st.markdown("#### 🔄 Churn Breakdown")
            fig, ax = plt.subplots(figsize=(8, 5))
            if col in num_cols:
                sns.boxplot(data=df, x='Churn', y=col, ax=ax)
            else:
                sns.countplot(data=df, x=col, hue='Churn', order=df[col].value_counts().index, ax=ax)
                plt.xticks(rotation=45)
            plt.title(f"{col} vs Churn")
            st.pyplot(fig)
    
    else:  # BIVARIATE ANALYSIS
        bi_type = st.selectbox("Bivariate Mode", ["Num vs Num", "Cat vs Cat", "Cat vs Num"])
        
        c1, c2 = st.columns(2)
        
        if bi_type == "Num vs Num":
            x_col = st.selectbox("X-Axis (Numerical)", num_cols)
            y_col = st.selectbox("Y-Axis (Numerical)", num_cols)
            
            with c1:
                st.markdown("#### 📍 Scatter View")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=df, x=x_col, y=y_col, hue='Churn', alpha=0.7, ax=ax)
                plt.title(f"{x_col} vs {y_col}")
                st.pyplot(fig)
            
            with c2:
                st.markdown("#### 📈 Trend Line")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.regplot(data=df, x=x_col, y=y_col, ax=ax)
                plt.title(f"{x_col} vs {y_col} Regression")
                st.pyplot(fig)
        
        elif bi_type == "Cat vs Cat":
            x_col = st.selectbox("Category Feature", cat_cols)
            
            with c1:
                st.markdown("#### 🔢 Count Overview")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df, x=x_col, order=df[x_col].value_counts().index, ax=ax)
                plt.xticks(rotation=45)
                plt.title(f"{x_col} Distribution")
                st.pyplot(fig)
            
            with c2:
                st.markdown("#### 🔄 Churn Split")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df, x=x_col, hue='Churn', order=df[x_col].value_counts().index, ax=ax)
                plt.xticks(rotation=45)
                plt.title(f"{x_col} by Churn")
                st.pyplot(fig)
        
        else:  # Cat vs Num
            cat_col = st.selectbox("Category Feature", cat_cols)
            num_col = st.selectbox("Numerical Feature", num_cols)
            
            with c1:
                st.markdown("#### 🧰 Box Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
                plt.xticks(rotation=45)
                plt.title(f"{cat_col} vs {num_col}")
                st.pyplot(fig)
            
            with c2:
                st.markdown("#### 🎻 Density View")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.violinplot(data=df, x=cat_col, y=num_col, ax=ax)
                plt.xticks(rotation=45)
                plt.title(f"{cat_col} vs {num_col} Violin")
                st.pyplot(fig)

# =====================================================
# ML PREDICTION SECTION
# =====================================================
elif section == "ml":
    st.markdown("Real-time churn risk assessment with enterprise-grade models.")
    
    model_name = st.selectbox("Model Algorithm", list(models_dict.keys()))
    model = models_dict[model_name]
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    # fpr, tpr, _ = roc_curve(y_test, y_prob)
    # roc_auc = auc(fpr, tpr)
    F1_score=f1_score(y_test, y_pred)
    precision=precision_score(y_test, y_pred)
    
    col1, col2 ,col3= st.columns(3)
    col1.metric("🎯 Accuracy", f"{accuracy:.2%}")
    col2.metric("📈 F1 Score", f"{F1_score:.3f}")
    col3.metric("📈 Precision", f"{precision:.2%}")

    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("#### 🔮 Customer Profile Input")
    with st.expander("Enter Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 24)
            monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 70)
            total_charges = st.number_input("Total Charges ($)", 0, 9000, 1500)
            contract = st.selectbox("Contract Type", df['Contract'].unique())
        
        with col2:
            internet = st.selectbox("Internet Service", df['InternetService'].unique())
            tech_support = st.selectbox("Tech Support", df['TechSupport'].unique())
            online_security = st.selectbox("Online Security", df['OnlineSecurity'].unique())
            payment = st.selectbox("Payment Method", df['PaymentMethod'].unique())
            paperless = st.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
    
    if st.button("🚀 Generate Prediction"):
        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "PaymentMethod": payment,
            "PaperlessBilling": paperless
        }
        
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
        input_scaled = scaler.transform(input_encoded)
        
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        
        risk_class = "high-risk" if pred == 1 else "low-risk"
        risk_label = "High Churn Risk" if pred == 1 else "Low Churn Risk"
        risk_prob = prob if pred == 1 else 1 - prob
        
        st.markdown(f"""
        <div class="prediction-card {risk_class}">
            <h3>{risk_label}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(risk_prob)
        st.info(f"Model: {model_name} | Overall Accuracy: {accuracy:.2%}")

# =====================================================
# INSIGHTS SECTION
# =====================================================
else:
    st.markdown("Data-driven recommendations to minimize customer churn.")
    
    st.markdown("#### 🔑 Core Findings")
    st.markdown("""
    - **Contract Dynamics**: Month-to-month plans see ~43% churn vs. ~3% for two-year commitments.
    - **Service Gaps**: Fiber optic users churn at ~42%; lack of Tech Support or Online Security significantly increases risk.
    - **Payment Patterns**: Electronic check payments are linked to the highest churn rate (~45%).
    - **Early Warning Signs**: Customers with low tenure and high monthly charges are most vulnerable.
    """)
    
    st.markdown("#### 🛡️ Recommended Retention Strategies")
    st.markdown("""
    - Offer attractive discounts or perks for upgrading to annual/two-year contracts.
    - Bundle Tech Support and Online Security services, especially for fiber optic customers.
    - Incentivize automatic payment methods (bank transfer/credit card) to reduce friction.
    - Implement proactive outreach for new customers and those with rising bills.
    """)
    
    st.success("These targeted actions can reduce churn by 15–25% based on industry benchmarks.")
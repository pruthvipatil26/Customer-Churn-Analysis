import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG & BASIC STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel Package Purchase Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f9fafb;}
    h1 {color: #1e40af;}
    .stButton>button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
    }
    .metric-box {
        background: white;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv("travel_dataset (2).csv")
    except FileNotFoundError:
        st.error("travel_dataset.csv not found in current directory!")
        st.stop()

    # Drop useless column if exists
    df = df.drop(columns=['CustomerID'], errors='ignore')

    # Basic cleaning
    df = df.drop_duplicates()

    # Normalize text columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # Gender normalization
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace({
            'fe male': 'female', 'femail': 'female', 'femal': 'female',
            'mail': 'male'
        })

    # Fill missing values
    df["TypeofContact"]=df["TypeofContact"].fillna(df["TypeofContact"].mode()[0])


    for col in df.select_dtypes(include=np.number):
        df[col] = df[col].fillna(df[col].median())

    # Convert appropriate numeric columns to int
    possible_int_cols = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
        'Passport', 'PitchSatisfactionScore', 'OwnCar',
        'NumberOfChildrenVisiting'
    ]

    for col in possible_int_cols:
        if col in df.columns:
            df[col] = df[col].round(0).astype(int)

    return df


# ──────────────────────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    # Target and features
    X = _df.drop('ProdTaken', axis=1)
    y = _df['ProdTaken']

    # Identify column types
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=0.22,
        random_state=42,
        stratify=y
    )

    # Scale numeric features (after encoding)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models with some imbalance handling
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_split=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }

    # Train all models
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

    # Return everything we need
    return (
        trained_models,
        scaler,
        X_encoded.columns.tolist(),   # ← very important: exact feature names after encoding
        X_test_scaled,
        y_test
    )


# ──────────────────────────────────────────────────────────────
# LOAD DATA & TRAIN MODELS
# ──────────────────────────────────────────────────────────────
df = load_and_prepare_data()

models, scaler, feature_columns, X_test_scaled, y_test = train_models(df)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# ──────────────────────────────────────────────────────────────
# MAIN INTERFACE
# ──────────────────────────────────────────────────────────────
st.title("✈️ Travel Package Purchase Predictor")
st.markdown("**Predict whether a customer will buy the travel package**")

page = st.sidebar.radio("Go to", ["EDA", "Model Performance", "Prediction"])

# ──────────────────────────────
# EDA Section
# ──────────────────────────────
if page == "EDA":
    st.header("Exploratory Data Analysis")

    selected_col = st.selectbox("Select column to analyze", df.columns)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        if selected_col in num_cols:
            sns.histplot(df[selected_col], kde=True, ax=ax)
        else:
            sns.countplot(x=df[selected_col], ax=ax)
            plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    with col2:
        st.subheader("vs Purchase")
        fig, ax = plt.subplots(figsize=(8, 5))
        if selected_col in num_cols:
            sns.boxplot(x='ProdTaken', y=selected_col, data=df, ax=ax)
        else:
            sns.countplot(x=selected_col, hue='ProdTaken', data=df, ax=ax)
            plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

# ──────────────────────────────
# Performance Section
# ──────────────────────────────
elif page == "Model Performance":
    st.header("Model Performance on Test Set")

    model_choice = st.selectbox("Select model", list(models.keys()))
    model = models[model_choice]

    y_pred = model.predict(X_test_scaled)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.2%}")
    col2.metric("Recall",    f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
    col3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2%}")
    col4.metric("F1 Score",  f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

# ──────────────────────────────
# PREDICTION SECTION - FIXED
# ──────────────────────────────
else:
    st.header("Predict New Customer")

    model_choice = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_choice]

    with st.form("customer_form"):
        st.subheader("Customer Information")

        input_data = {}
        input_columns = st.columns(3)

        for i, col in enumerate(df.drop("ProdTaken", axis=1).columns):
            with input_columns[i % 3]:
                if col in num_cols:
                    default = int(df[col].median())
                    input_data[col] = st.number_input(
                        col,
                        value=default,
                        step=1,
                        format="%d"
                    )
                else:
                    unique_vals = sorted(df[col].unique())
                    input_data[col] = st.selectbox(
                        col,
                        options=unique_vals,
                        index=0
                    )

        submitted = st.form_submit_button("Make Prediction", use_container_width=True)

    if submitted:
        # 1. Create DataFrame with original columns
        input_df = pd.DataFrame([input_data])

        # 2. Make sure ALL original columns exist
        original_cols = df.drop("ProdTaken", axis=1).columns
        for col in original_cols:
            if col not in input_df.columns:
                if col in num_cols:
                    input_df[col] = int(df[col].median())
                else:
                    input_df[col] = df[col].mode()[0]

        # 3. Apply the SAME encoding as in training
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # 4. Align columns exactly with training data
        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # 5. Scale
        input_scaled = scaler.transform(input_encoded)

        # 6. Predict
        prob = model.predict_proba(input_scaled)[0][1]
        pred_class = 1 if prob >= 0.38 else 0   # adjustable threshold

        st.markdown("### Prediction Result")

        if pred_class == 1:
            st.success(f"**HIGH PROBABILITY OF PURCHASE**  \nProbability: **{prob:.1%}**")
            st.progress(prob)
        else:
            st.warning(f"**LOW PROBABILITY OF PURCHASE**  \nProbability: **{prob:.1%}**")
            st.progress(prob)

        st.caption("Note: threshold set to 0.38 to better detect potential buyers")

st.markdown("---")
st.caption("Travel Package Purchase Prediction • Streamlit • 2025–2026")
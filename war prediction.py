pip install streamlit pandas numpy scikit-learn


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
data_path = 'UcdpPrioConflict_v24_1.csv'
df = pd.read_csv(data_path)

# Preprocess data (simplified for dashboard)
def preprocess_data(df):
    """
    Preprocess the dataset by dropping irrelevant columns, handling missing values, and encoding categorical data.
    """
    # Drop unnecessary columns
    columns_to_drop = [
        "side_a_2nd", "side_b_2nd", "ep_end_prec", "territory_name", 
        "start_date", "start_date2", "ep_end_date", "version"
    ]
    df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Fill missing values
    numeric_cols = df_cleaned.select_dtypes(include=[float, int]).columns
    categorical_cols = df_cleaned.select_dtypes(include=[object]).columns

    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna(df_cleaned[categorical_cols].mode().iloc[0])

    # Encode categorical columns
    df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

    # Add target variables
    if "intensity_level" in df_encoded.columns:
        df_encoded["high_intensity"] = (df_encoded["intensity_level"] > 1).astype(int)
    if "ep_end" in df_encoded.columns:
        df_encoded["duration"] = df_encoded["ep_end"]  # Proxy for duration

    return df_encoded

# Preprocess the dataset
df_encoded = preprocess_data(df)

# Feature and target selection
X = df_encoded.drop(columns=[col for col in ["intensity_level", "high_intensity", "duration"] if col in df_encoded.columns], errors='ignore')
y_classification = df_encoded["high_intensity"] if "high_intensity" in df_encoded.columns else None
y_regression = df_encoded["duration"] if "duration" in df_encoded.columns else None

# Train-test split
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42) if y_classification is not None else (None, None, None, None)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42) if y_regression is not None else (None, None, None, None)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) if X_train is not None else None
X_test_scaled = scaler.transform(X_test) if X_test is not None else None

# Train models
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=20)
if y_train_class is not None:
    rf_classifier.fit(X_train_scaled, y_train_class)

rf_regressor = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20)
if y_train_reg is not None:
    rf_regressor.fit(X_train_scaled, y_train_reg)

# Streamlit Dashboard
st.title("Conflict Prediction Dashboard")
st.write("Predict the likelihood and duration of conflicts based on user input.")

# Sidebar for input features
st.sidebar.header("Input Features")
user_input = {}

if X is not None:
    for feature in X.columns:
        user_input[feature] = st.sidebar.number_input(
            f"{feature}", 
            min_value=float(X[feature].min()), 
            max_value=float(X[feature].max()), 
            value=float(X[feature].median())
        )

    # Convert user input to DataFrame and scale
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    # Predictions
    if y_classification is not None:
        classification_pred = rf_classifier.predict(input_scaled)[0]
        classification_prob = rf_classifier.predict_proba(input_scaled)[0, 1]
        st.subheader("Classification Prediction")
        st.write(f"**Conflict Likelihood:** {'High-Intensity Conflict' if classification_pred == 1 else 'Low-Intensity Conflict'}")
        st.write(f"**Probability of High-Intensity Conflict:** {classification_prob:.2f}")

    if y_regression is not None:
        regression_pred = rf_regressor.predict(input_scaled)[0]
        st.subheader("Regression Prediction")
        st.write(f"**Estimated Conflict Duration:** {regression_pred:.2f} years")

    # Feature Importance Visualization
    if y_classification is not None:
        st.subheader("Feature Importance")
        feature_importances = rf_classifier.feature_importances_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances}).sort_values(by="Importance", ascending=False)
        st.bar_chart(importance_df.set_index("Feature").head(10))

st.write("This dashboard provides predictions based on input features for conflict analysis and planning.")


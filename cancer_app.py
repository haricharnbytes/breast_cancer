import streamlit as st
import joblib
import os
import pandas as pd
import pickle

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction System")
st.markdown("Adjust sliders or type values directly to predict cancer diagnosis.")

# ----------------------------------
# Load Feature Names
# ----------------------------------

@st.cache_resource
def load_features():
    with open("features.pkl", "rb") as f:
        return pickle.load(f)


features = load_features()

if isinstance(features, dict):
    feature_names = list(features.keys())
else:
    feature_names = features

# ----------------------------------
# Load Feature Ranges and Means
# ----------------------------------

@st.cache_resource
def load_feature_stats():
    ranges = joblib.load("feature_ranges.pkl")
    means = joblib.load("feature_means.pkl")
    return ranges, means


feature_ranges, feature_means = load_feature_stats()

# ----------------------------------
# Load Models
# ----------------------------------

MODEL_DIR = "saved_models"

model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

if len(model_files) == 0:
    st.error("❌ No trained models found")
    st.stop()

# ----------------------------------
# Sidebar Model Selection
# ----------------------------------

st.sidebar.header("⚙ Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    model_files
)

model_path = os.path.join(MODEL_DIR, selected_model_name)

@st.cache_resource
def load_model(path):
    return joblib.load(path)


model = load_model(model_path)

st.sidebar.success("✅ Model Loaded")

# ----------------------------------
# Feature Input UI (Slider + Box)
# ----------------------------------

st.subheader("📊 Feature Input")

input_data = {}

col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

for i, feature in enumerate(feature_names):

    with cols[i % 3]:

        min_val, max_val = feature_ranges[feature]
        default_val = feature_means[feature]

        step_val = (max_val - min_val) / 100

        # Create two columns for slider + input box
        slider_col, input_col = st.columns([3, 1])

        with slider_col:
            slider_value = st.slider(
                label=feature,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=float(step_val),
                key=f"{feature}_slider"
            )

        with input_col:
            input_value = st.number_input(
                label="",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(slider_value),
                step=float(step_val),
                key=f"{feature}_input"
            )

        # Sync slider and input box
        final_value = input_value

        input_data[feature] = final_value


input_df = pd.DataFrame([input_data])

# ----------------------------------
# Prediction
# ----------------------------------

if st.button("🔍 Predict Diagnosis"):

    try:

        prediction = model.predict(input_df)

        st.success("✅ Prediction Completed")

        class_map = {
            0: "Benign",
            1: "Malignant"
        }

        predicted_class = prediction[0]

        st.subheader("📊 Prediction Result")

        if predicted_class in class_map:
            st.metric("Diagnosis", class_map[predicted_class])
        else:
            st.metric("Prediction", predicted_class)

        # ----------------------------------
        # Probability Output
        # ----------------------------------

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(input_df)

            prob_df = pd.DataFrame(
                probs,
                columns=model.classes_
            )

            st.subheader("📈 Prediction Confidence")

            st.dataframe(prob_df)
            st.bar_chart(prob_df.T)

    except Exception as e:

        st.error("❌ Prediction Failed")
        st.exception(e)

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")
st.caption("Breast Cancer Prediction App | Slider + Numeric Input UI")

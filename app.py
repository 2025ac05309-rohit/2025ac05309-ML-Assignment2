import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

from model.preprocessing import preprocess_data
from model.evaluation import evaluate_model

from model.logistic_regression import logistic_regression_model
from model.decision_tree import decision_tree_model
from model.knn import knn_model
from model.naive_bayes import naive_bayes_model
from model.random_forest import random_forest_model


# Page Configuration
st.set_page_config(
    page_title="Dry Bean Classification System",
    page_icon="🌱",
    layout="wide"
)

# Sidebar
st.sidebar.title("Assignment Information")
st.sidebar.markdown("""
### Machine Learning Assignment 2

**Dataset**
- Dry Bean Dataset

**Problem Type**
- Multi-Class Classification

**Algorithms**
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Random Forest
""")
st.sidebar.success("Train Dataset Loaded Successfully")


# Main Heading
st.title("🌱 Dry Bean Classification System")

st.markdown("""
This application performs **multi-class classification** on the **Dry Bean Dataset**
using five supervised machine learning algorithms.

Upload the provided **test_data.csv**, choose a classification model,
and evaluate its performance.
""")

st.divider()


# Dataset Information
st.subheader("Dataset Information")
col1, col2, col3 = st.columns(3)
with col1: st.info("**Dataset**\n\nDry Bean Dataset")

with col2: st.info("**Classes**\n\n7")

with col3: st.info("**Features**\n\n16")

st.divider()

st.subheader("⚙️ Model Evaluation")

left, right = st.columns([1, 1])
# Upload 
with left:
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload test data",
            type=["csv"],
        )
        if uploaded_file is not None:
            st.success(f"✅ {uploaded_file.name}")
# Model Selection
# Run
with right:
    with st.container(border=True):
        selected_model = st.selectbox(
            "Select model",
            [
                "Logistic Regression",
                "Decision Tree",
                "K-Nearest Neighbors",
                "Gaussian Naive Bayes",
                "Random Forest"
            ],
        )
        run_model = st.button(
            "🚀 Evaluate Model",
            width="stretch"
        )

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    with st.expander("📄 Preview Uploaded Dataset"):
        st.dataframe(
            test_df.head(),
            width="stretch"
        )
        st.caption(
            f"Rows: {test_df.shape[0]} | Columns: {test_df.shape[1]}"
        )
st.divider()


# Run Selected Model
if run_model:
    if uploaded_file is None:
        st.warning("Please upload test_data.csv before continuing.")
        st.stop()

    progress = st.progress(0, text="Loading dataset...")
    # Prepare Training Data
    (X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test,scaler,label_encoder) = preprocess_data(save_test_csv=False)

    progress.progress(25, text="Reading uploaded dataset...")
    X_uploaded = test_df.drop("Class", axis=1)
    y_uploaded = label_encoder.transform(test_df["Class"])

    progress.progress(50, text="Training selected model...")

    # Select Model
    if selected_model == "Logistic Regression":
        model = logistic_regression_model(X_train_scaled,y_train)
        X_test_final = scaler.transform(X_uploaded)

    elif selected_model == "Decision Tree":
        model = decision_tree_model(X_train,y_train)
        X_test_final = X_uploaded

    elif selected_model == "K-Nearest Neighbors":
        model = knn_model(X_train_scaled,y_train)
        X_test_final = scaler.transform(X_uploaded)

    elif selected_model == "Gaussian Naive Bayes":
        model = naive_bayes_model(X_train,y_train)
        X_test_final = X_uploaded

    else:
        model = random_forest_model(X_train,y_train)
        X_test_final = X_uploaded

    progress.progress(75, text="Evaluating model...")
    # Evaluate
    results = evaluate_model(model,X_test_final,y_uploaded,label_encoder)
    progress.progress(100, text="Completed!")
    time.sleep(0.5)
    progress.empty()

    st.success("Model evaluation completed successfully!")

    result_df = pd.DataFrame([{
        "Model": selected_model,
        "Accuracy": results["Accuracy"],
        "AUC": results["AUC"],
        "Precision": results["Precision"],
        "Recall": results["Recall"],
        "F1": results["F1"],
        "MCC": results["MCC"]
    }])

    st.download_button(
        "📥 Download Results",
        result_df.to_csv(index=False),
        file_name="evaluation_results.csv",
        mime="text/csv"
    )
    
    # Evaluation Metrics
    st.divider()

    st.subheader("📊 Evaluation Metrics")
    st.info(f"Selected Model: **{selected_model}**")
    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Accuracy", results["Accuracy"])
    metric2.metric("AUC", results["AUC"])
    metric3.metric("Precision", results["Precision"])

    metric4, metric5, metric6 = st.columns(3)

    metric4.metric("Recall", results["Recall"])
    metric5.metric("F1 Score", results["F1"])
    metric6.metric("MCC", results["MCC"])

    st.divider()

    left, right = st.columns([1, 1.5])

    with left:

        st.subheader("📋 Confusion Matrix")

        fig, ax = plt.subplots(figsize=(4, 4))

        cm = results["Confusion Matrix"]

        image = ax.imshow(cm, cmap="Greens")

        ax.set_xticks(range(len(label_encoder.classes_)))
        ax.set_yticks(range(len(label_encoder.classes_)))

        ax.set_xticklabels(
            label_encoder.classes_,
            rotation=45,
            ha="right",
            fontsize=8
        )

        ax.set_yticklabels(
            label_encoder.classes_,
            fontsize=8
        )

        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

        # Cell values
        threshold = cm.max() / 2

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > threshold else "black"
                )

        plt.tight_layout()

        st.pyplot(fig)

        st.caption("Rows = Actual • Columns = Predicted")

    with right:
        st.subheader("📑 Classification Report")

        report_df = (
            pd.DataFrame(results["Classification Report"])
            .transpose()
            .round(3)
        )

        st.dataframe(
            report_df,
            width="stretch",
            height="content"
        )
    
    # Model Summary
    st.divider()
    st.subheader("🏆 Model Summary")
    accuracy = results["Accuracy"]
    if accuracy >= 0.95:
        performance = "Excellent"
        message = "The selected model achieved excellent performance on the uploaded test dataset."
    elif accuracy >= 0.90:
        performance = "Very Good"
        message = "The selected model achieved very good classification performance."
    elif accuracy >= 0.80:
        performance = "Good"
        message = "The selected model performed well, with scope for further improvement."
    else:
        performance = "Average"
        message = "The selected model produced acceptable results but may require further tuning."

    st.success(
        f"""
        **Selected Model:** {selected_model}
        
        **Performance:** {performance}
        
        {message}
        """
    )
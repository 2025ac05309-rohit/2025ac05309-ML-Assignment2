import pandas as pd
import streamlit as st

from model.evaluation import evaluate_model
from model.preprocessing import preprocess_data

from model.logistic_regression import logistic_regression_model
from model.decision_tree import decision_tree_model
from model.knn import knn_model
from model.naive_bayes import naive_bayes_model
from model.random_forest import random_forest_model

@st.cache_resource
def main():
    (X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test,scaler,label_encoder) = preprocess_data()
    
    results = []

    # Logistic Regression
    print("\n" + "=" * 60)
    print("Running Logistic Regression")
    print("=" * 60)
    model = logistic_regression_model(X_train_scaled,y_train)
    metrics = evaluate_model(model,X_test_scaled,y_test,label_encoder)
    metrics["Model"] = "Logistic Regression"
    results.append(metrics)

    # Decision Tree
    print("\n" + "=" * 60)
    print("Running Decision Tree")
    print("=" * 60)
    model = decision_tree_model(X_train,y_train)
    metrics = evaluate_model(model,X_test,y_test,label_encoder)
    metrics["Model"] = "Decision Tree"
    results.append(metrics)

    # KNN
    print("\n" + "=" * 60)
    print("Running KNN")
    print("=" * 60)
    model = knn_model(X_train_scaled,y_train)
    metrics = evaluate_model(model,X_test_scaled,y_test,label_encoder)
    metrics["Model"] = "KNN"
    results.append(metrics)

    # Gaussian Naive Bayes
    print("\n" + "=" * 60)
    print("Running Gaussian Naive Bayes")
    print("=" * 60)
    model = naive_bayes_model(X_train,y_train)
    metrics = evaluate_model(model,X_test,y_test,label_encoder)
    metrics["Model"] = "Gaussian Naive Bayes"
    results.append(metrics)

    # Random Forest
    print("\n" + "=" * 60)
    print("Running Random Forest")
    print("=" * 60)
    model = random_forest_model(X_train,y_train)
    metrics = evaluate_model(model,X_test,y_test,label_encoder)
    metrics["Model"] = "Random Forest"
    results.append(metrics)

    # Comparison Table
    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model","Accuracy","AUC","Precision","Recall","F1","MCC"]
    ]

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results_df)

    results_df.to_csv("results.csv", index=False)
    print("\nresults.csv saved successfully.")


if __name__ == "__main__":
    main()
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test, label_encoder):
    # Predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test,y_prob,multi_class="ovr",average="weighted")
    precision = precision_score(y_test,y_pred,average="weighted")
    recall = recall_score(y_test,y_pred,average="weighted")
    f1 = f1_score(y_test,y_pred,average="weighted")
    mcc = matthews_corrcoef(y_test,y_pred)

    # Display Results
    print("\nPerformance Metrics")
    print("----------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC Score: {mcc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print("----------------------")
    print(classification_report(y_test,y_pred,target_names=label_encoder.classes_))

    return {
        "Accuracy": accuracy,
        "AUC Score": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC Score": mcc
    }
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

    return {
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        ),
        "Predictions": y_pred
    }
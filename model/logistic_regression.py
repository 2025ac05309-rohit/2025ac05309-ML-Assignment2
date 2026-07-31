from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess_data
from evaluation import evaluate_model

def logistic_regression_model():
    (X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder) = preprocess_data()

    # Create Model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)

    # Train model
    model.fit(X_train_scaled, y_train)
    print("Model training completed")

    # Extract results
    results = evaluate_model(model,X_test_scaled,y_test,label_encoder)

    return model, results

if __name__ == "__main__":
    logistic_regression_model()

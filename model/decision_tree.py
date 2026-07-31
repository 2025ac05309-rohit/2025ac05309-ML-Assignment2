from sklearn.tree import DecisionTreeClassifier

from preprocessing import preprocess_data
from evaluation import evaluate_model

def decision_tree_model():
    (X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test,scaler,label_encoder) = preprocess_data()

    # Create Model
    print("\nTraining Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42)

    # Train Model
    model.fit(X_train, y_train)
    print("Model trained successfully")

    # Evaluate Model
    results = evaluate_model(model,X_test,y_test,label_encoder)

    return model, results


if __name__ == "__main__":
    decision_tree_model()
    
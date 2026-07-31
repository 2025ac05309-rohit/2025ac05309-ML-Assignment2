from sklearn.neighbors import KNeighborsClassifier

from preprocessing import preprocess_data
from evaluation import evaluate_model


def knn_model():
    (X_train,X_test,X_train_scaled,X_test_scaled,y_train,y_test,scaler,label_encoder) = preprocess_data()

    # Create Model
    print("\nTraining K-Nearest Neighbors Classifier...")
    model = KNeighborsClassifier(n_neighbors=5)

    # Train Model
    model.fit(X_train_scaled, y_train)
    print("Model trained successfully")

    # Evaluate Model
    results = evaluate_model(model,X_test_scaled,y_test,label_encoder)

    return model, results

if __name__ == "__main__":
    knn_model()

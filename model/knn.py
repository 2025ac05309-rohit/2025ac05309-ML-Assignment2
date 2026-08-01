from sklearn.neighbors import KNeighborsClassifier

def knn_model(X_train_scaled,y_train):
    
    # Create Model
    print("\nTraining K-Nearest Neighbors Classifier...")
    model = KNeighborsClassifier(n_neighbors=5)

    # Train Model
    model.fit(X_train_scaled, y_train)
    print("Model trained successfully")

    return model

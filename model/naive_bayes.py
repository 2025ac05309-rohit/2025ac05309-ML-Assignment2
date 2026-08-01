from sklearn.naive_bayes import GaussianNB

def naive_bayes_model(X_train,y_train):
    
    # Create Model
    print("\nTraining Gaussian Naive Bayes Classifier...")
    model = GaussianNB()

    # Train Model
    model.fit(X_train, y_train)
    print("Model trained successfully")

    return model

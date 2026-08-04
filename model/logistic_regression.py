from sklearn.linear_model import LogisticRegression

def logistic_regression_model(X_train_scaled,y_train):

    # Create Model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)

    # Train model
    model.fit(X_train_scaled, y_train)
    print("Logistic Regression model training completed")

    return model

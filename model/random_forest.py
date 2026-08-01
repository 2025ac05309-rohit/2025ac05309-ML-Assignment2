from sklearn.ensemble import RandomForestClassifier

def random_forest_model(X_train,y_train):
    
    # Create Model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100,random_state=42)

    # Train Model
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    return model
    
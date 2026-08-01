from sklearn.tree import DecisionTreeClassifier

def decision_tree_model(X_train,y_train):
    
    # Create Model
    print("\nTraining Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42)

    # Train Model
    model.fit(X_train, y_train)
    print("Model trained successfully")

    return model

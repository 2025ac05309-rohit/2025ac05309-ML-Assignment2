import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path


def preprocess_data(file_path="./data/Dry_Bean_Dataset.csv",test_set_size=0.20,seed=42,save_test_csv=True):
    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / "data" / "Dry_Bean_Dataset.csv"
    # Load dataset
    print("Reading dataset...")
    df = pd.read_csv(file_path)
    print(f"Original Shape: {df.shape}")

    # Remove duplicate rows
    print("Finding duplicates...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Duplicate Rows: {duplicates}\nDropping duplicates...")
        df = df.drop_duplicates()
        print(f"Shape After Removing Duplicates: {df.shape}")
    else: 
        print("No duplicates found")

    # Separate Features and target
    print("Separating features and target...")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Encode target labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print("\nClass Encoding:")
    print("----------------------")
    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name} --> {index}")

    # Train-test split
    print(f"\nSplitting data into {int((1 - test_set_size) * 100)}% training and {int(test_set_size * 100)}% testing set...")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_set_size,random_state=seed,stratify=y)

    if save_test_csv:
        print("Saving test_data.csv...")
        test_data = X_test.copy()
        test_data["Class"] = label_encoder.inverse_transform(y_test)
        test_data.to_csv("./test_data.csv", index=False)
        print("test_data.csv created successfully")
    
    # Feature scaling
    print("Performing feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Final Shapes
    print("\nFinal Dataset Summary")
    print("----------------------")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")
    
    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        label_encoder
    )
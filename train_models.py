import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Ensure datasets directory exists
data_dir = "datasets"
if not os.path.exists(data_dir):
    raise FileNotFoundError(
        f"The directory '{data_dir}' does not exist. Make sure you have created it and placed the dataset files inside.")


# Function to check if a file exists
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file '{file_path}' not found. Make sure the file exists in the correct location.")


# Function to preprocess data
def preprocess_data(df, target_column):
    # Drop duplicate rows if any
    df.drop_duplicates(inplace=True)

    # Fill missing values with the column median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Convert categorical columns (if any) to numeric using Label Encoding
    for col in df.columns:
        if df[col].dtype == "object":
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Function to train and save model
def train_and_save_model(data_path, target_column, model_filename):
    check_file_exists(data_path)
    df = pd.read_csv(data_path)

    # Preprocess the data
    X, y = preprocess_data(df, target_column)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_filename} Accuracy: {accuracy:.2f}")

    # Ensure models directory exists
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {model_path}")


# Train models
train_and_save_model(os.path.join(data_dir, "diabetes.csv"), "Outcome", "diabetes_model.sav")
train_and_save_model(os.path.join(data_dir, "heart.csv"), "target", "heart_model.sav")
train_and_save_model(os.path.join(data_dir, "parkinsons.csv"), "status", "parkinsons_model.sav")
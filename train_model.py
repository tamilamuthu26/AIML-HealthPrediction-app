import os
from django.conf import settings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load dataset
data = pd.read_csv('diabetes.csv')  # Ensure your CSV file is in the same directory
#url = 'https://www.kaggle.com/code/farzadnekouei/heart-disease-prediction'


# Define features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model trained and saved successfully.")




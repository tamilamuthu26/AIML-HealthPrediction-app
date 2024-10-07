import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Load data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Train the chatbot model
def train_chatbot():
    # Load the nutrition data
    file_path = os.path.join(os.path.dirname(__file__),'nutrition_data.csv')
    data = load_data(file_path)

    # Prepare the data
    X = data['question']
    y = data['answer']

    # Create a TF-IDF vectorizer and fit it to the questions
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Train a classifier
    model = MultinomialNB()
    model.fit(X_vectorized, y)

    # Save the model and vectorizer
    with open('chatbot_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_chatbot()

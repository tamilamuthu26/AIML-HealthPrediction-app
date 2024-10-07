import json
from pyexpat import model
from django.shortcuts import render
from django.http import HttpResponseBadRequest, JsonResponse
import pickle
import numpy as np
from sklearn import datasets
from torch import cosine_similarity

# Load the trained model and scaler
with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
def home(request):
    return render(request, 'home.html')

def predict_diabetes(request):
    if request.method == 'POST':
        # Get data from the form
        pregnancies = float(request.POST['pregnancies'])
        glucose = float(request.POST['glucose'])
        blood_pressure = float(request.POST['blood_pressure'])
        skin_thickness = float(request.POST['skin_thickness'])
        insulin = float(request.POST['insulin'])
        bmi = float(request.POST['bmi'])
        diabetes_pedigree_function = float(request.POST['diabetes_pedigree_function'])
        age = float(request.POST['age'])

        # Prepare input data for prediction
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = knn_model.predict(input_data_scaled)
        
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        return render(request, 'result.html', {'result': result})

    return render(request, 'predict.html')

# diabetes/views.py
from django.shortcuts import render, redirect
from django.http import HttpResponse

# Existing imports...

def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        # Here you can process the contact form data, like sending an email or saving to the database.
        return HttpResponse(f"Thank you for your message, {name}!")

    return render(request, 'diabetes/home.html')  # Redirect back to home or show a contact success page

def services(request):
    return render(request, 'services.html')

def predict_heart_disease(request):
    return render(request, 'heart.html')


# heart/views.py



# # Load the trained model
# model_path = os.path.join(settings.BASE_DIR,  'knn_model_heart.pkl')
# model = joblib.load(model_path)

# from django.shortcuts import render
# import joblib
# import os
# import numpy as np
# from django.conf import settings

# # Load the trained KNN model and scaler
# model_path = os.path.join(settings.BASE_DIR,  'knn_model_heart.pkl')
# scaler_path = os.path.join(settings.BASE_DIR,  'scaler_heart.pkl')

# knn_model = joblib.load(model_path)
# scaler = joblib.load(scaler_path)

# def heart_disease_form(request):
#     result = None
    
#     if request.method == "POST":
#         # Debugging: Print the received POST data
#         print("Received POST data:", request.POST)

#         age = int(request.POST.get('age'))
#         gender = int(request.POST.get('gender'))
#         cholesterol = int(request.POST.get('cholesterol'))
#         bp = int(request.POST.get('bp'))
#         max_heart_rate = int(request.POST.get('max_heart_rate'))
#         exercise_angina = int(request.POST.get('exercise_angina'))
#         st_depression = float(request.POST.get('st_depression'))

#         # Prepare the input data for prediction
#         input_data = np.array([[age, gender, cholesterol, bp, max_heart_rate, exercise_angina, st_depression]])

#         # Scale the input data
#         input_data_scaled = scaler.transform(input_data)

#         # Make the prediction
#         prediction = knn_model.predict(input_data_scaled)
#         result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"

#         # Debugging: Print the prediction result
#         print("Prediction result:", result)

#     return render(request, 'heartResult.html', {'result': result})



# views.py

# #Global variable to store the dataset
# import csv
# import os

# # Function to load nutrition data from CSV file
# def load_nutrition_data():
#     # Construct the correct path to the CSV file
#     file_path = os.path.join(os.path.dirname(__file__), 'nutrition', 'nutrition_data.csv')  # Adjust the path as needed
    
#     # Check if file exists at the path
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"CSV file not found at path: {file_path}")
    
#     nutrition_data = []
    
#     # Open the CSV file and read its contents
#     with open(file_path, mode='r') as file:
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
#             nutrition_data.append({
#                 'question': row['question'],
#                 'answer': row['answer']
#             })
    
#     return nutrition_data


# def chatbot_view(request):
#     if request.method == 'POST':
#         user_input = request.POST.get('user_input').lower()
        
#         # Default bot response if no match is found
#         bot_response = "I'm not sure how to respond to that. Can you ask something else?"
        
#         # Check the user's input against the dataset
#         for item in datasets:
#             if item['question'].lower() in user_input:
#                 bot_response = item['answer']
#                 break

#         return JsonResponse({'response': bot_response})

#     return render(request, 'chatbot.html')
# import csv
# import os
# from fuzzywuzzy import process
# from django.http import JsonResponse
# from django.shortcuts import render

# # Load the nutrition data once at module level
# nutrition_data = []

# def load_nutrition_data():
#     file_path = os.path.join(os.path.dirname(__file__), 'data','nutrition', 'nutrition_data.csv')
    
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"CSV file not found at path: {file_path}")

#     with open(file_path, mode='r') as file:
#         csv_reader = csv.DictReader(file)
#         for row in csv_reader:
#             nutrition_data.append({
#                 'question': row['question'].lower(),  # Store questions in lowercase for easier matching
#                 'answer': row['answer']
#             })

# # Call the function to load data when the module is loaded
# load_nutrition_data()

# def chatbot_view(request):
#     if request.method == 'POST':
#         user_input = request.POST.get('user_input').lower()
        
#         # Debugging: Print user input
#         print(f"User input: {user_input}")

#         # Default bot response if no match is found
#         bot_response = "I'm not sure how to respond to that. Can you ask something else?"

#         # Extract the questions from the nutrition data for fuzzy matching
#         questions = [item['question'] for item in nutrition_data]

#         # Get the best match using fuzzy matching
#         best_match, score = process.extractOne(user_input, questions)

#         # Set a threshold for the score to consider it a valid match
#         if score > 70:  # Adjust the threshold as needed
#             # Find the answer corresponding to the best match
#             index = questions.index(best_match)
#             bot_response = nutrition_data[index]['answer']

#         # Debugging: Print bot response
#         print(f"Bot response: {bot_response}")

#         return JsonResponse({'response': bot_response})

#     return render(request, 'chatbot.html')
import os
import pickle
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd

# Load the trained model and vectorizer
def load_chatbot_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '..', 'nutrition', 'chatbot_model.pkl')
    vectorizer_path = os.path.join(base_dir, '..', 'nutrition', 'vectorizer.pkl')

    # Load the chatbot model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the vectorizer
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return vectorizer, model

# Load the dataset
def load_nutrition_data():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'nutrition', 'nutrition_data.csv')  # Adjust the path as needed
    nutrition_data = pd.read_csv(file_path)  # Read the CSV file
    return nutrition_data

# Load model, vectorizer, and nutrition data globally
vectorizer, model = load_chatbot_model()
nutrition_data = load_nutrition_data()

def chatbot_view(request):
    if request.method == 'POST':
        # Load the nutrition data and model
        nutrition_data = load_nutrition_data()
        vectorizer, model = load_chatbot_model()
        
        # Get user input
        user_input = json.loads(request.body).get('user_input')

        # Transform user input using the vectorizer
        user_input_vec = vectorizer.transform([user_input])  # Use transform instead of np.vectorize
        prediction = model.predict(user_input_vec)

        # Prepare the bot's response
        bot_response = prediction[0]  # Get the predicted response

        return JsonResponse({'response': bot_response})

    return JsonResponse({'response': 'Invalid request method.'})
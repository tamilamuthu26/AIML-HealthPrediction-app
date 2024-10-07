"""
URL configuration for healthPrediction project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path

from diabetes import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', views.predict_diabetes, name='predict_diabetes'),
    path('', views.home, name='home'),  # Add this line for the root URL
    path('contact/', views.contact, name='contact'),
    path('services/', views.services, name='services'), 
          # Path for the prediction form
    
  path('chatbot/', views.chatbot_view, name='chatbot'),
   path('chatbot/', views.chatbot_view, name='chatbot_view'),
    path('heart/', views.predict_heart_disease, name='predict_heart_disease'),  # Path for the result
]


    


   


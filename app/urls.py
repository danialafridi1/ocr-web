
from django.urls import path
from .views import askAi, index

urlpatterns = [
    path('', index, name='index'),  # root URL shows index
    path('ask_ai/',askAi,name='askAi')
]
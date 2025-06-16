
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('process_image/', views.process_image, name='process_image'),
    path('summarize_result/', views.summarize_result, name='summarize_result'),
    path('ask_ai/', views.askAi, name='askAi') # OCR page
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


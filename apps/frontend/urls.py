from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('coming-soon/', views.coming_soon, name='coming_soon'),
    path('auth/', views.signupin, name='signupin'),
    path('api/subscribe/', views.subscribe_to_mailchimp, name='subscribe'),

    # Placeholder URLs for authentication (to be implemented)
    path('login/', views.home, name='login'),
    path('register/', views.home, name='register'),
    path('password-reset/', views.home, name='password_reset'),
]

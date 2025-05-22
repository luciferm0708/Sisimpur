from django.urls import path
from . import views

urlpatterns = [
<<<<<<< HEAD
    path('', views.home, name='home'),
    path('coming-soon/', views.coming_soon, name='coming_soon'),
    path('api/subscribe/', views.subscribe_to_mailchimp, name='subscribe'),
    path('submit/', views.submit_form, name='submit_form'),
=======
    path("", views.home, name="home"),
    path("coming-soon/", views.coming_soon, name="coming_soon"),
    path("api/subscribe/", views.subscribe_to_mailchimp, name="subscribe"),
>>>>>>> 3783834637fcca91e4f3b3f00f2c1bc310b3f7a2
]

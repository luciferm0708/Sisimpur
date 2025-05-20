from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('frontend.urls')),
    path('auth/', include('authentication.urls')),
    path("__reload__/", include("django_browser_reload.urls")),  # Add django-browser-reload URLs
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

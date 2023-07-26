from django.urls import path, include

urlpatterns = [
    path('stock/', include('myapp.urls')),
    
]

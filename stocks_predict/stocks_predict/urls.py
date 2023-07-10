from django.urls import path, include

urlpatterns = [
    path('stock/', include('stocks.urls')),
]

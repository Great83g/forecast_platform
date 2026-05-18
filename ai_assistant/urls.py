from django.urls import path

from . import views

app_name = "ai_assistant"

urlpatterns = [
    path("query/", views.assistant_query, name="query"),
]

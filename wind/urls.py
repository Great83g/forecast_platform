from django.urls import path

from . import views

app_name = "wind"

urlpatterns = [
    path("", views.station_list, name="station-list"),
    path("station/create/", views.station_create, name="station-create"),
]

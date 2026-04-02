from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from stations.models import Organization, Station
from .models import WindStationProfile


class WindPagesTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="wind_user", password="password123")
        self.org = Organization.objects.create(name="Wind Org", owner=self.user)

    def test_list_requires_auth(self):
        response = self.client.get(reverse("wind:station-list"))
        self.assertEqual(response.status_code, 302)

    def test_create_wind_station(self):
        self.client.force_login(self.user)
        payload = {
            "name": "Wind Farm 1",
            "org": self.org.id,
            "latitude": 50.0,
            "longitude": 70.0,
            "timezone": "Asia/Almaty",
            "data_shift_hours": 0,
            "turbine_count": 5,
            "turbine_rated_power_kw": 3000,
            "hub_height_m": 105,
            "rotor_diameter_m": 130,
            "cut_in_speed_ms": 3,
            "rated_speed_ms": 12,
            "cut_out_speed_ms": 25,
        }

        response = self.client.post(reverse("wind:station-create"), data=payload)

        self.assertEqual(response.status_code, 302)
        station = Station.objects.get(name="Wind Farm 1")
        self.assertEqual(station.station_kind, Station.KIND_WIND)
        self.assertEqual(station.capacity_mw, 15.0)
        profile = WindStationProfile.objects.get(station=station)
        self.assertEqual(profile.turbine_count, 5)

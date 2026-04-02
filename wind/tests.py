from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
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


class WindModuleRouteTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="wind_user2", password="password123")
        self.org = Organization.objects.create(name="Wind Org 2", owner=self.user)
        self.station = Station.objects.create(
            org=self.org,
            name="Wind Route Station",
            station_kind=Station.KIND_WIND,
            capacity_mw=4.2,
            capacity_ac_kw=4200,
            capacity_dc_kw=4200,
        )

    def test_wind_station_action_pages_return_200(self):
        self.client.force_login(self.user)

        urls = [
            reverse("wind:station-detail", args=[self.station.pk]),
            reverse("wind:station-upload", args=[self.station.pk]),
            reverse("wind:station-forecast-list", args=[self.station.pk]),
            reverse("wind:station-train", args=[self.station.pk]),
        ]
        for url in urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)

    def test_wind_station_list_uses_wind_module_urls(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("wind:station-list"))

        self.assertContains(response, reverse("wind:station-detail", args=[self.station.pk]))
        self.assertContains(response, reverse("wind:station-upload", args=[self.station.pk]))
        self.assertContains(response, reverse("wind:station-forecast-list", args=[self.station.pk]))
        self.assertContains(response, reverse("wind:station-train", args=[self.station.pk]))


class WindHistoryUploadTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="wind_user3", password="password123")
        self.org = Organization.objects.create(name="Wind Org 3", owner=self.user)
        self.station = Station.objects.create(
            org=self.org,
            name="Wind Upload Station",
            station_kind=Station.KIND_WIND,
            capacity_mw=2.0,
            capacity_ac_kw=2000,
            capacity_dc_kw=2000,
        )

    def test_upload_csv_creates_wind_history_records(self):
        from .models import WindRecord

        self.client.force_login(self.user)
        csv_payload = (
            "ds,power_kw,wind_speed_ms,wind_direction_deg,air_temp,air_density\n"
            "2026-03-01 00:00:00,100,5.1,180,12,1.20\n"
            "2026-03-01 01:00:00,120,5.5,190,11,1.19\n"
        ).encode("utf-8")
        upload = SimpleUploadedFile("wind_history.csv", csv_payload, content_type="text/csv")

        response = self.client.post(
            reverse("wind:station-upload", args=[self.station.pk]),
            data={"action": "upload", "history_scope": "main", "file": upload},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(WindRecord.objects.filter(station=self.station, history_scope="main").count(), 2)

    def test_export_history_returns_excel(self):
        from .models import WindRecord
        from django.utils import timezone

        self.client.force_login(self.user)
        WindRecord.objects.create(
            station=self.station,
            history_scope="main",
            timestamp=timezone.now(),
            power_kw=111.0,
            wind_speed_ms=4.4,
        )

        response = self.client.get(reverse("wind:station-export-history", args=[self.station.pk]))

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", response["Content-Type"])


class WindForecastModuleTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="wind_user4", password="password123")
        self.org = Organization.objects.create(name="Wind Org 4", owner=self.user)
        self.station = Station.objects.create(
            org=self.org,
            name="Wind Forecast Station",
            station_kind=Station.KIND_WIND,
            capacity_mw=4.2,
            capacity_ac_kw=4200,
            capacity_dc_kw=4200,
            latitude=48.0,
            longitude=67.5,
            timezone="Asia/Almaty",
        )

    @patch("wind.views.fetch_visual_crossing_hourly")
    @patch("wind.views.fetch_open_meteo_hourly")
    def test_forecast_run_creates_rows_for_scope(self, om_mock, vc_mock):
        import pandas as pd
        from types import SimpleNamespace
        from .models import WindForecast

        self.client.force_login(self.user)
        df = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2026-04-03 06:00:00", "2026-04-03 07:00:00"]),
                "air_temp": [12.0, 11.0],
                "wind_speed": [7.0, 8.0],
                "cloudcover": [20.0, 30.0],
                "humidity": [50.0, 55.0],
                "precip": [0.0, 0.1],
            }
        )
        vc_mock.return_value = SimpleNamespace(ok=True, source="visual_crossing", df=df, error=None)
        om_mock.return_value = SimpleNamespace(ok=False, source="open_meteo", df=pd.DataFrame(), error="disabled")

        response = self.client.get(
            reverse("wind:station-forecast-run", args=[self.station.pk]),
            {"days": "2", "scope": "test", "providers": ["visual_crossing"]},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(WindForecast.objects.filter(station=self.station, forecast_scope="test").count(), 2)


    @patch("wind.views.send_report_email")
    @patch("wind.views.fetch_visual_crossing_hourly")
    def test_forecast_run_auto_send_email_calls_mailer(self, vc_mock, send_mock):
        import pandas as pd
        from types import SimpleNamespace

        self.client.force_login(self.user)
        df = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2026-04-03 06:00:00"]),
                "air_temp": [12.0],
                "wind_speed": [7.0],
                "cloudcover": [20.0],
                "humidity": [50.0],
                "precip": [0.0],
            }
        )
        vc_mock.return_value = SimpleNamespace(ok=True, source="visual_crossing", df=df, error=None)
        send_mock.return_value = True

        response = self.client.get(
            reverse("wind:station-forecast-run", args=[self.station.pk]),
            {
                "days": "1",
                "scope": "test",
                "providers": ["visual_crossing"],
                "emails": "a@test.com,b@test.com",
                "auto_send": "1",
            },
        )

        self.assertEqual(response.status_code, 302)
        send_mock.assert_called_once()


    @patch("wind.views.send_report_email")
    @patch("wind.views.fetch_visual_crossing_hourly")
    def test_forecast_run_email_failure_does_not_crash(self, vc_mock, send_mock):
        import pandas as pd
        from types import SimpleNamespace

        self.client.force_login(self.user)
        df = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2026-04-03 06:00:00"]),
                "air_temp": [12.0],
                "wind_speed": [7.0],
                "cloudcover": [20.0],
                "humidity": [50.0],
                "precip": [0.0],
            }
        )
        vc_mock.return_value = SimpleNamespace(ok=True, source="visual_crossing", df=df, error=None)
        send_mock.side_effect = Exception("smtp down")

        response = self.client.get(
            reverse("wind:station-forecast-run", args=[self.station.pk]),
            {
                "days": "1",
                "scope": "test",
                "providers": ["visual_crossing"],
                "emails": "a@test.com",
                "auto_send": "1",
            },
        )

        self.assertEqual(response.status_code, 302)

    def test_forecast_list_page_works(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("wind:station-forecast-list", args=[self.station.pk]))
        self.assertEqual(response.status_code, 200)


    def test_forecast_export_returns_excel_for_tz_aware_timestamps(self):
        from .models import WindForecast
        from django.utils import timezone

        self.client.force_login(self.user)
        WindForecast.objects.create(
            station=self.station,
            forecast_scope="test",
            timestamp=timezone.now(),
            pred_heur=100.0,
            pred_final=100.0,
            weather_source="visual_crossing",
        )

        response = self.client.get(reverse("wind:station-forecast-export", args=[self.station.pk]), {"scope": "test"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", response["Content-Type"])

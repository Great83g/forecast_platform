from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from dashboard.models import ForecastSchedule
from dashboard.services.forecast_scheduler import run_scheduled_forecasts
from stations.models import Organization, Station
from .models import WindRecord, WindStationProfile


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
            "auto_history_enabled": "on",
            "auto_history_folder": "/mnt/share/wind/farm1",
            "auto_history_script": "example_wind",
            "auto_history_run_time": "06:30",
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
        self.assertTrue(station.auto_history_enabled)
        self.assertEqual(station.auto_history_folder, "/mnt/share/wind/farm1")
        self.assertEqual(station.auto_history_script, "example_wind")
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
            reverse("wind:station-edit", args=[self.station.pk]),
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
        self.assertContains(response, reverse("wind:station-edit", args=[self.station.pk]))
        self.assertContains(response, reverse("wind:station-upload", args=[self.station.pk]))
        self.assertContains(response, reverse("wind:station-forecast-list", args=[self.station.pk]))
        self.assertContains(response, reverse("wind:station-train", args=[self.station.pk]))

    def test_wind_station_edit_updates_auto_history_fields(self):
        from .models import WindStationProfile

        self.client.force_login(self.user)
        WindStationProfile.objects.create(
            station=self.station,
            turbine_count=2,
            turbine_rated_power_kw=2000,
            hub_height_m=100,
            rotor_diameter_m=120,
            cut_in_speed_ms=3,
            rated_speed_ms=12,
            cut_out_speed_ms=25,
        )
        payload = {
            "name": self.station.name,
            "org": self.org.id,
            "latitude": 48.0,
            "longitude": 67.5,
            "timezone": "Asia/Almaty",
            "data_shift_hours": 1,
            "auto_history_enabled": "on",
            "auto_history_folder": "/mnt/share/wind/edit",
            "auto_history_script": "example_wind",
            "auto_history_run_time": "07:15",
            "turbine_count": 2,
            "turbine_rated_power_kw": 2000,
            "hub_height_m": 100,
            "rotor_diameter_m": 120,
            "cut_in_speed_ms": 3,
            "rated_speed_ms": 12,
            "cut_out_speed_ms": 25,
        }

        response = self.client.post(reverse("wind:station-edit", args=[self.station.pk]), data=payload)

        self.assertEqual(response.status_code, 302)
        self.station.refresh_from_db()
        self.assertTrue(self.station.auto_history_enabled)
        self.assertEqual(self.station.auto_history_folder, "/mnt/share/wind/edit")
        self.assertEqual(self.station.auto_history_script, "example_wind")


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

    @patch("wind.views.fetch_visual_crossing_hourly")
    @patch("wind.views.fetch_open_meteo_hourly")
    def test_manual_forecast_runs_preserve_previous_target_dates(self, om_mock, vc_mock):
        import pandas as pd
        from types import SimpleNamespace
        from .models import WindForecast, WindForecastRun

        self.client.force_login(self.user)
        om_mock.return_value = SimpleNamespace(ok=False, source="open_meteo", df=pd.DataFrame(), error="disabled")
        vc_mock.side_effect = [
            SimpleNamespace(
                ok=True,
                source="visual_crossing",
                df=pd.DataFrame(
                    {
                        "ds": pd.to_datetime(["2026-05-14 00:00:00", "2026-05-14 01:00:00"]),
                        "air_temp": [12.0, 11.0],
                        "wind_speed": [7.0, 8.0],
                        "cloudcover": [20.0, 30.0],
                        "humidity": [50.0, 55.0],
                        "precip": [0.0, 0.1],
                    }
                ),
                error=None,
            ),
            SimpleNamespace(
                ok=True,
                source="visual_crossing",
                df=pd.DataFrame(
                    {
                        "ds": pd.to_datetime(["2026-05-15 00:00:00", "2026-05-15 01:00:00"]),
                        "air_temp": [13.0, 12.0],
                        "wind_speed": [9.0, 10.0],
                        "cloudcover": [25.0, 35.0],
                        "humidity": [51.0, 56.0],
                        "precip": [0.0, 0.0],
                    }
                ),
                error=None,
            ),
        ]

        for _ in range(2):
            response = self.client.get(
                reverse("wind:station-forecast-run", args=[self.station.pk]),
                {"days": "1", "scope": "main", "providers": ["visual_crossing"]},
            )
            self.assertEqual(response.status_code, 302)

        self.assertEqual(WindForecastRun.objects.filter(station=self.station, forecast_scope="main").count(), 2)
        self.assertEqual(WindForecast.objects.filter(station=self.station, forecast_scope="main").count(), 4)
        self.assertEqual(
            WindForecast.objects.filter(station=self.station, forecast_scope="main", timestamp__date="2026-05-14").count(),
            2,
        )
        self.assertEqual(
            WindForecast.objects.filter(station=self.station, forecast_scope="main", timestamp__date="2026-05-15").count(),
            2,
        )

        response_14 = self.client.get(
            reverse("wind:station-detail", args=[self.station.pk]),
            {"date_from": "14.05.2026", "date_to": "14.05.2026"},
        )
        response_15 = self.client.get(
            reverse("wind:station-detail", args=[self.station.pk]),
            {"date_from": "15.05.2026", "date_to": "15.05.2026"},
        )

        self.assertEqual(response_14.status_code, 200)
        self.assertEqual(response_15.status_code, 200)
        self.assertEqual(response_14.context["points_count"], 2)
        self.assertEqual(response_15.context["points_count"], 2)



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


    def test_schedule_update_redirects_to_main_scope(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("wind:station-forecast-schedule-update", args=[self.station.pk]),
            data={
                "enabled": "on",
                "run_time": "06:00",
                "days": 2,
                "providers": ["visual_crossing"],
                "emails": "a@test.com",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("scope=main", response["Location"])

    def test_forecast_list_page_works(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("wind:station-forecast-list", args=[self.station.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "manual-scope-select")
        self.assertContains(response, "manual-horizon-mode")


    def test_forecast_export_returns_excel_for_tz_aware_timestamps(self):
        from .models import WindForecast, WindForecastRun
        from django.utils import timezone

        self.client.force_login(self.user)
        run = WindForecastRun.objects.create(
            station=self.station,
            forecast_scope="test",
            forecast_base_date=timezone.localdate(),
            provider="visual_crossing",
            horizon_days=1,
        )
        WindForecast.objects.create(
            station=self.station,
            forecast_run=run,
            forecast_scope="test",
            timestamp=timezone.now(),
            pred_heur=100.0,
            pred_final=100.0,
            weather_source="visual_crossing",
        )

        response = self.client.get(reverse("wind:station-forecast-export", args=[self.station.pk]), {"scope": "test"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", response["Content-Type"])

    @patch("dashboard.services.forecast_scheduler.send_report_email")
    @patch("dashboard.services.forecast_scheduler.fetch_weather_for_wind")
    def test_wind_scheduler_runs_without_email(self, weather_mock, send_mock):
        import pandas as pd
        from .models import WindForecast

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
        weather_mock.return_value = (df, "visual_crossing", [])

        ForecastSchedule.objects.create(
            station=self.station,
            enabled=True,
            run_time="06:00",
            days=2,
            horizon_mode="legacy",
            providers="visual_crossing",
            emails="",
        )

        count = run_scheduled_forecasts(force=True)

        self.assertEqual(count, 1)
        self.assertEqual(WindForecast.objects.filter(station=self.station, forecast_scope="main").count(), 2)
        send_mock.assert_not_called()


class WindAutoHistoryServiceTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="wind_auto_user", password="password123")
        self.org = Organization.objects.create(name="Wind Auto Org", owner=self.user)
        self.station = Station.objects.create(
            org=self.org,
            name="Wind Auto Station",
            station_kind=Station.KIND_WIND,
            capacity_mw=2.0,
            capacity_ac_kw=2000,
            capacity_dc_kw=2000,
            auto_history_enabled=True,
        )

    def test_dashboard_auto_history_uses_wind_upsert_for_wind_station(self):
        from dashboard.services.history_autofill import upsert_station_history_from_share

        with TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "wind.csv").write_text(
                "ds,power_kw,wind_speed_ms,air_temp\n"
                "2026-03-01 00:10:00,100,5.1,11\n"
                "2026-03-01 01:20:00,120,5.5,12\n",
                encoding="utf-8",
            )
            self.station.auto_history_folder = str(folder)
            self.station.save(update_fields=["auto_history_folder"])

            rows = upsert_station_history_from_share(self.station)

        self.assertEqual(rows, 2)
        self.assertEqual(WindRecord.objects.filter(station=self.station, history_scope=WindRecord.HISTORY_SCOPE_MAIN).count(), 2)

    def test_wind_auto_history_custom_script_name_works(self):
        from wind.services.history_autofill import upsert_station_history_from_share

        with TemporaryDirectory() as td:
            self.station.auto_history_folder = td
            self.station.auto_history_script = "example_wind"
            self.station.save(update_fields=["auto_history_folder", "auto_history_script"])

            rows = upsert_station_history_from_share(self.station)

        self.assertEqual(rows, 0)

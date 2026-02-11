from django.contrib.auth.models import User
from django.test import RequestFactory, TestCase
from django.utils import timezone
from unittest.mock import patch

from dashboard.models import ForecastSchedule
from dashboard.services.forecast_engine import run_forecast_for_station
from dashboard.services.forecast_scheduler import run_scheduled_forecasts
from dashboard.views import station_forecast_scheduler_tick
from stations.models import Organization, Station


class ForecastEngineIndexRegressionTests(TestCase):
    def setUp(self):
        user = User.objects.create_user(username="tester", password="pass")
        org = Organization.objects.create(name="Org", owner=user)
        self.station = Station.objects.create(
            org=org,
            name="SES 8.8MW",
            capacity_mw=8.8,
            capacity_ac_kw=8800,
            capacity_dc_kw=9000,
            latitude=None,
            longitude=None,
        )

    @patch("dashboard.services.forecast_engine._target_offsets_for_weekday_calendar", return_value=[2])
    def test_run_forecast_handles_sparse_index_after_target_date_filter(self, _offsets):
        """
        Regression: после фильтрации target_dates DataFrame feat может иметь индекс,
        начинающийся не с 0. Сохранение прогноза должно работать без IndexError.
        """
        result = run_forecast_for_station(
            station_id=self.station.pk,
            days=1,
            use_models=False,
            horizon_mode="weekday_calendar",
        )

        self.assertTrue(result["ok"])
        self.assertGreater(result["count"], 0)


class ForecastSchedulerForceRunTests(TestCase):
    def setUp(self):
        user = User.objects.create_user(username="scheduler", password="pass")
        org = Organization.objects.create(name="Scheduler Org", owner=user)
        self.station = Station.objects.create(
            org=org,
            name="Scheduler Station",
            capacity_mw=1.0,
            capacity_ac_kw=1000,
            capacity_dc_kw=1100,
        )
        self.schedule = ForecastSchedule.objects.create(
            station=self.station,
            enabled=True,
            days=1,
            run_time=timezone.datetime.strptime("23:59", "%H:%M").time(),
            emails="test@example.com",
        )

    @patch("dashboard.services.forecast_scheduler.send_report_email")
    @patch("dashboard.services.forecast_scheduler.build_forecast_report", return_value=object())
    @patch(
        "dashboard.services.forecast_scheduler.run_forecast_for_station",
        return_value={"ok": True, "weather_source": "stub"},
    )
    def test_force_run_ignores_time_and_last_run_limit(self, _run, _build, _send):
        now = timezone.now().replace(hour=9, minute=0, second=0, microsecond=0)

        first = run_scheduled_forecasts(now=now, force=True)
        second = run_scheduled_forecasts(now=now, force=True)

        self.assertEqual(first, 1)
        self.assertEqual(second, 1)


    @patch("dashboard.services.forecast_scheduler.send_report_email")
    @patch("dashboard.services.forecast_scheduler.build_forecast_report", return_value=object())
    @patch(
        "dashboard.services.forecast_scheduler.run_forecast_for_station",
        return_value={"ok": True, "weather_source": "stub", "days": 3},
    )
    def test_report_uses_effective_days_from_engine_result(self, _run, build_mock, send_mock):
        now = timezone.now().replace(hour=23, minute=59, second=0, microsecond=0)

        count = run_scheduled_forecasts(now=now, force=True)

        self.assertEqual(count, 1)
        build_mock.assert_called_once()
        self.assertEqual(build_mock.call_args.kwargs["days"], 3)
        send_mock.assert_called_once()
        self.assertEqual(send_mock.call_args.args[3], 3)

    @patch("dashboard.services.forecast_scheduler.send_report_email")
    @patch("dashboard.services.forecast_scheduler.build_forecast_report", return_value=object())
    @patch(
        "dashboard.services.forecast_scheduler.run_forecast_for_station",
        return_value={"ok": True, "weather_source": "stub"},
    )
    def test_non_force_respects_daily_limit(self, _run, _build, _send):
        now = timezone.now().replace(hour=23, minute=59, second=0, microsecond=0)

        first = run_scheduled_forecasts(now=now, force=False)
        second = run_scheduled_forecasts(now=now, force=False)

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)


class ForecastSchedulerTickViewTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(username="viewer", password="pass")

    @patch("dashboard.views.run_scheduled_forecasts", return_value=3)
    def test_scheduler_tick_parses_force_true(self, run_mock):
        request = self.factory.get("/dashboard/stations/1/forecast/scheduler-tick/?force=1")
        request.user = self.user

        response = station_forecast_scheduler_tick(request)

        self.assertEqual(response.status_code, 200)
        run_mock.assert_called_once_with(force=True)
        self.assertIn(b'"force": true', response.content)

    @patch("dashboard.views.run_scheduled_forecasts", return_value=0)
    def test_scheduler_tick_defaults_force_false(self, run_mock):
        request = self.factory.get("/dashboard/stations/1/forecast/scheduler-tick/")
        request.user = self.user

        response = station_forecast_scheduler_tick(request)

        self.assertEqual(response.status_code, 200)
        run_mock.assert_called_once_with(force=False)
        self.assertIn(b'"force": false', response.content)

from django.contrib.auth.models import User
from django.test import RequestFactory, TestCase
from django.utils import timezone
import pandas as pd
from unittest.mock import patch

from dashboard.models import ForecastSchedule
from dashboard.services.forecast_engine import _target_offsets_for_weekday_calendar, run_forecast_for_station
from dashboard.services.forecast_scheduler import _normalize_schedule_providers, run_scheduled_forecasts
from dashboard.views import _parse_history_datetime, station_forecast_scheduler_tick
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


class ForecastEngineWeekdayCalendarTests(TestCase):
    def test_friday_offsets_cover_exactly_three_next_days(self):
        friday = timezone.datetime(2026, 2, 13, 9, 0)

        offsets = _target_offsets_for_weekday_calendar(friday)

        self.assertEqual(offsets, [1, 2, 3])

    def test_weekday_mode_report_days_do_not_depend_on_requested_days(self):
        user = User.objects.create_user(username="daysmode", password="pass")
        org = Organization.objects.create(name="Days Mode Org", owner=user)
        station = Station.objects.create(
            org=org,
            name="Days Mode Station",
            capacity_mw=1.0,
            capacity_ac_kw=1000,
            capacity_dc_kw=1100,
            latitude=None,
            longitude=None,
        )

        with patch("dashboard.services.forecast_engine._target_offsets_for_weekday_calendar", return_value=[1, 2, 3]):
            result_days_1 = run_forecast_for_station(
                station_id=station.pk,
                days=1,
                use_models=False,
                horizon_mode="weekday_calendar",
            )
            result_days_7 = run_forecast_for_station(
                station_id=station.pk,
                days=7,
                use_models=False,
                horizon_mode="weekday_calendar",
            )

        self.assertTrue(result_days_1["ok"])
        self.assertTrue(result_days_7["ok"])
        self.assertEqual(result_days_1["days"], 3)
        self.assertEqual(result_days_7["days"], 3)


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
    def test_scheduler_uses_main_scope(self, run_mock, build_mock, _send):
        now = timezone.now().replace(hour=23, minute=59, second=0, microsecond=0)

        count = run_scheduled_forecasts(now=now, force=True)

        self.assertEqual(count, 1)
        self.assertEqual(run_mock.call_args.kwargs["forecast_scope"], "main")
        self.assertEqual(build_mock.call_args.kwargs["forecast_scope"], "main")

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
        return_value={"ok": True, "weather_source": "stub", "days": 3, "target_dates": ["2026-02-15", "2026-02-16", "2026-02-17"]},
    )
    def test_report_uses_effective_days_from_engine_result(self, _run, build_mock, send_mock):
        now = timezone.now().replace(hour=23, minute=59, second=0, microsecond=0)

        count = run_scheduled_forecasts(now=now, force=True)

        self.assertEqual(count, 1)
        build_mock.assert_called_once()
        self.assertEqual(build_mock.call_args.kwargs["days"], 3)
        self.assertEqual(
            build_mock.call_args.kwargs["target_dates"],
            ["2026-02-15", "2026-02-16", "2026-02-17"],
        )
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

    @patch("dashboard.services.forecast_scheduler.send_report_email")
    @patch("dashboard.services.forecast_scheduler.build_forecast_report", return_value=object())
    @patch(
        "dashboard.services.forecast_scheduler.run_forecast_for_station",
        return_value={"ok": True, "weather_source": "stub", "days": 3},
    )
    def test_schedule_open_meteo_only_disables_models(self, run_mock, _build, _send):
        self.schedule.providers = "visual_crossing,open_meteo_only"
        self.schedule.save(update_fields=["providers"])

        now = timezone.now().replace(hour=23, minute=59, second=0, microsecond=0)
        count = run_scheduled_forecasts(now=now, force=True)

        self.assertEqual(count, 1)
        self.assertEqual(run_mock.call_args.kwargs["providers"], ["open_meteo"])
        self.assertFalse(run_mock.call_args.kwargs["use_models"])


class ForecastSchedulerProviderNormalizationTests(TestCase):
    def test_open_meteo_only_provider_marker_forces_open_meteo_and_heuristic(self):
        providers, open_meteo_only = _normalize_schedule_providers("visual_crossing,open_meteo_only")

        self.assertEqual(providers, ["open_meteo"])
        self.assertTrue(open_meteo_only)


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


class HistoryDatetimeParsingTests(TestCase):
    def test_prefers_day_first_for_ambiguous_dates(self):
        series = pd.Series(["01/03/2026 10:00:00", "02/03/2026 09:00:00"])

        parsed = _parse_history_datetime(series)

        self.assertEqual(parsed.iloc[0].month, 3)
        self.assertEqual(parsed.iloc[0].day, 1)
        self.assertEqual(parsed.iloc[1].month, 3)
        self.assertEqual(parsed.iloc[1].day, 2)

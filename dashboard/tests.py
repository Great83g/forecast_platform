from datetime import time
from types import SimpleNamespace

from django.contrib.auth.models import User
from django.test import RequestFactory, TestCase
from django.utils import timezone
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch

from dashboard.models import ForecastSchedule
from dashboard.services.forecast_engine import _target_offsets_for_weekday_calendar, run_forecast_for_station
from dashboard.services.forecast_scheduler import _normalize_schedule_providers, run_scheduled_forecasts
from dashboard.views import _parse_history_datetime, station_forecast_scheduler_tick
from dashboard.services.history_autofill import (
    _resolve_station_share_folder,
    collect_share_history_dataframe,
    run_auto_history_updates,
    upsert_station_history_from_share,
)

from dashboard.forms import StationForm
from dashboard.management.commands.run_scheduled_forecasts import _run_auto_history_updates_safe
from solar.models import SolarRecord
from stations.models import Organization, OrganizationMember, Station


def build_custom_history_dataframe(station):
    now = timezone.now().replace(minute=0, second=0, microsecond=0)
    return pd.DataFrame(
        [
            {
                "ds": pd.Timestamp(now),
                "irradiation": 500.1,
                "air_temp": 20.2,
                "pv_temp": 24.3,
                "power_kw": 700.4,
            }
        ]
    )



class StationFormAutoHistoryFolderInitialTests(TestCase):
    def test_edit_form_shows_station_specific_folder_when_model_has_default_share(self):
        user = User.objects.create_user(username="folder-form", password="pass")
        org = Organization.objects.create(name="Folder Form Org", owner=user)
        station = Station.objects.create(
            org=org,
            name="SES 8.8 MW",
            capacity_mw=8.8,
            auto_history_folder="/mnt/share",
        )

        form = StationForm(instance=station, user=user)

        self.assertEqual(form["auto_history_folder"].value(), f"/mnt/share/org_{org.id}/SES_8.8_MW")



class StationEditAutoHistoryFolderNormalizationTests(TestCase):
    def test_edit_get_normalizes_plain_share_folder_for_existing_station(self):
        user = User.objects.create_user(username="folder-edit", password="pass")
        org = Organization.objects.create(name="Folder Edit Org", owner=user)
        OrganizationMember.objects.create(
            organization=org,
            user=user,
            role=OrganizationMember.ROLE_OWNER,
        )

        station = Station.objects.create(
            org=org,
            name="SES 8.8 MW",
            capacity_mw=8.8,
        )
        Station.objects.filter(pk=station.pk).update(auto_history_folder="/mnt/share")

        self.client.login(username="folder-edit", password="pass")
        response = self.client.get(f"/dashboard/station/{station.pk}/edit/")

        self.assertEqual(response.status_code, 200)
        station.refresh_from_db()
        self.assertEqual(station.auto_history_folder, f"/mnt/share/org_{org.id}/SES_8.8_MW")
    def test_edit_form_prefers_tmp_path_when_org_has_tmp_station(self):
        user = User.objects.create_user(username="folder-form-tmp", password="pass")
        org = Organization.objects.create(name="Folder Form Tmp Org", owner=user)

        Station.objects.create(
            org=org,
            name="SES 1.2 MW",
            capacity_mw=1.2,
            auto_history_folder=f"/tmp/forecast_platform_auto_history/org_{org.id}/SES_1.2_MW",
        )
        station = Station.objects.create(
            org=org,
            name="SES 8.8 MW",
            capacity_mw=8.8,
            auto_history_folder="/mnt/share",
        )

        form = StationForm(instance=station, user=user)

        self.assertEqual(
            form["auto_history_folder"].value(),
            f"/tmp/forecast_platform_auto_history/org_{org.id}/SES_8.8_MW",
        )





class StationAutoHistoryFolderFallbackTests(TestCase):
    def test_missing_station_subfolder_falls_back_to_share_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            station = SimpleNamespace(auto_history_folder=str(base / "org_1" / "SES_1.2_MW"), pk=42)

            resolved = _resolve_station_share_folder(station, share_root=base)

            self.assertEqual(resolved, base)

    def test_non_share_path_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            station = SimpleNamespace(auto_history_folder=str(base / "other" / "path"), pk=43)

            resolved = _resolve_station_share_folder(station, share_root=base / "share")

            self.assertEqual(resolved, Path(station.auto_history_folder))



class StationAutoHistoryCustomScriptTests(TestCase):
    def setUp(self):
        user = User.objects.create_user(username="autohistory-script", password="pass")
        org = Organization.objects.create(name="AutoHistory Script Org", owner=user)
        self.station = Station.objects.create(
            org=org,
            name="Script Station",
            capacity_mw=1.0,
            auto_history_enabled=True,
            auto_history_script="dashboard.tests:build_custom_history_dataframe",
        )

    def test_upsert_uses_station_custom_script(self):
        rows = upsert_station_history_from_share(self.station)

        self.assertEqual(rows, 1)
        rec = SolarRecord.objects.get(station=self.station)
        self.assertAlmostEqual(rec.power_kw, 700.4)

    def test_short_module_name_resolves_from_history_scripts_package(self):
        self.station.auto_history_script = "example_station"

        rows = upsert_station_history_from_share(self.station)

        self.assertEqual(rows, 1)

    def test_file_path_module_can_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "custom_builder.py"
            fpath.write_text(
                "import pandas as pd\n"
                "def build_history_dataframe(station):\n"
                "    return pd.DataFrame([{\n"
                "        'ds': pd.Timestamp('2026-01-01 10:00:00'),\n"
                "        'irradiation': 400.0,\n"
                "        'air_temp': 15.0,\n"
                "        'pv_temp': 20.0,\n"
                "        'power_kw': 500.0,\n"
                "    }])\n"
            )
            self.station.auto_history_script = f"{fpath}:build_history_dataframe"

            rows = upsert_station_history_from_share(self.station)

        self.assertEqual(rows, 1)

    def test_invalid_custom_script_format_falls_back_to_standard_handler(self):
        self.station.auto_history_script = ":build_history_dataframe"

        rows = upsert_station_history_from_share(self.station)

        self.assertEqual(rows, 0)

    def test_script_value_with_help_text_uses_first_token(self):
        self.station.auto_history_script = "example_station или dashboard.services.history_scripts.example_station:build_history_dataframe"

        rows = upsert_station_history_from_share(self.station)

        self.assertEqual(rows, 1)



class StationAutoHistoryScheduleTests(TestCase):
    def setUp(self):
        user = User.objects.create_user(username="autohistory-time", password="pass")
        org = Organization.objects.create(name="AutoHistory Time Org", owner=user)
        self.station = Station.objects.create(
            org=org,
            name="Timed Script Station",
            capacity_mw=1.0,
            auto_history_enabled=True,
            auto_history_script="dashboard.tests:build_custom_history_dataframe",
            auto_history_run_time=time(6, 0),
        )

    @patch("dashboard.services.history_autofill.timezone.localtime")
    def test_run_auto_history_updates_skips_before_station_time(self, localtime_mock):
        localtime_mock.return_value = timezone.datetime(2026, 2, 20, 5, 30, tzinfo=timezone.get_current_timezone())

        rows = run_auto_history_updates()

        self.assertEqual(rows, 0)

    @patch("dashboard.services.history_autofill.timezone.localtime")
    def test_run_auto_history_updates_runs_once_per_day(self, localtime_mock):
        localtime_mock.return_value = timezone.datetime(2026, 2, 20, 6, 30, tzinfo=timezone.get_current_timezone())

        first_rows = run_auto_history_updates()
        second_rows = run_auto_history_updates()

        self.assertEqual(first_rows, 1)
        self.assertEqual(second_rows, 0)
        self.station.refresh_from_db()
        self.assertEqual(str(self.station.auto_history_last_run_date), "2026-02-20")

    @patch("dashboard.services.history_autofill._safe_upsert_station", return_value=(1, True))
    @patch("dashboard.services.history_autofill.timezone.localtime")
    def test_run_auto_history_updates_allows_near_time_grace_window(self, localtime_mock, upsert_mock):
        self.station.auto_history_last_run_date = None
        self.station.auto_history_run_time = time(9, 20)
        self.station.save(update_fields=["auto_history_last_run_date", "auto_history_run_time"])
        localtime_mock.return_value = timezone.datetime(2026, 2, 24, 9, 19, 31, tzinfo=timezone.get_current_timezone())

        rows = run_auto_history_updates()

        self.assertEqual(rows, 1)
        upsert_mock.assert_called_once()

    @patch("dashboard.services.history_autofill._safe_upsert_station", return_value=(1, True))
    @patch("dashboard.services.history_autofill.timezone.localtime")
    def test_run_auto_history_updates_allows_pre_time_run_when_scheduler_is_sparse(self, localtime_mock, upsert_mock):
        self.station.auto_history_last_run_date = timezone.datetime(2026, 2, 23).date()
        self.station.auto_history_run_time = time(9, 0)
        self.station.save(update_fields=["auto_history_last_run_date", "auto_history_run_time"])
        localtime_mock.return_value = timezone.datetime(2026, 2, 24, 6, 0, tzinfo=timezone.get_current_timezone())

        rows = run_auto_history_updates()

        self.assertEqual(rows, 1)
        upsert_mock.assert_called_once()
        self.station.refresh_from_db()
        self.assertEqual(str(self.station.auto_history_last_run_date), "2026-02-24")


    @patch("dashboard.services.history_autofill._safe_upsert_station", side_effect=[(0, True), (1, True)])
    @patch("dashboard.services.history_autofill.timezone.localtime")
    def test_run_auto_history_updates_retries_same_day_when_no_rows(self, localtime_mock, _safe_upsert_mock):
        localtime_mock.return_value = timezone.datetime(2026, 2, 20, 6, 30, tzinfo=timezone.get_current_timezone())

        first_rows = run_auto_history_updates()
        second_rows = run_auto_history_updates()

        self.assertEqual(first_rows, 0)
        self.assertEqual(second_rows, 1)
        self.station.refresh_from_db()
        self.assertEqual(str(self.station.auto_history_last_run_date), "2026-02-20")



class StationAutoHistoryConfigChangeResetTests(TestCase):
    def test_station_save_resets_last_run_date_when_auto_history_config_changes(self):
        user = User.objects.create_user(username="autohistory-reset", password="pass")
        org = Organization.objects.create(name="AutoHistory Reset Org", owner=user)
        station = Station.objects.create(
            org=org,
            name="Reset Station",
            capacity_mw=1.0,
            auto_history_enabled=True,
            auto_history_run_time=time(6, 0),
            auto_history_last_run_date=timezone.datetime(2026, 2, 20).date(),
        )

        station.auto_history_run_time = time(7, 0)
        station.save()

        station.refresh_from_db()
        self.assertIsNone(station.auto_history_last_run_date)



class StationAutoHistoryMergeSameDateTests(TestCase):
    @patch("dashboard.services.history_autofill.read_meteo_hourly")
    @patch("dashboard.services.history_autofill.read_plant_report_hourly")
    def test_collect_share_history_merges_two_reports_for_same_date(self, plant_mock, meteo_mock):
        meteo_mock.return_value = pd.DataFrame(
            [{"ds": pd.Timestamp("2026-02-17 10:00:00"), "irradiation": 500, "air_temp": 20, "pv_temp": 25}]
        )
        plant_mock.side_effect = [
            pd.DataFrame([{"ds": pd.Timestamp("2026-02-17 10:00:00"), "power_kw": 100.0}]),
            pd.DataFrame([{"ds": pd.Timestamp("2026-02-17 10:00:00"), "power_kw": 30.0}]),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "D222152_20260217_0000.csv.gz").write_text("x")
            (folder / "Plant Report_SPP 1.2 MW_17-02-2026_part1.xlsx").write_text("x")
            (folder / "Plant Report_SPP 1.2 MW_17-02-2026_part2.xlsx").write_text("x")

            out = collect_share_history_dataframe(folder)

        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["power_kw"]), 130.0)



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


class StationOrderingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="order", password="pass")
        self.org = Organization.objects.create(name="Order Org", owner=self.user)
        self.client.login(username="order", password="pass")

        self.station_a = Station.objects.create(org=self.org, name="A")
        self.station_b = Station.objects.create(org=self.org, name="B")
        self.station_c = Station.objects.create(org=self.org, name="C")

    def test_station_list_sorted_by_sort_order(self):
        names = list(
            Station.objects.filter(org=self.org)
            .order_by("sort_order", "id")
            .values_list("name", flat=True)
        )
        self.assertEqual(names, ["A", "B", "C"])

    def test_move_station_down_swaps_with_next_station(self):
        response = self.client.post(f"/dashboard/station/{self.station_a.pk}/move/down/")

        self.assertEqual(response.status_code, 302)
        names = list(
            Station.objects.filter(org=self.org)
            .order_by("sort_order", "id")
            .values_list("name", flat=True)
        )
        self.assertEqual(names, ["B", "A", "C"])

    def test_move_station_up_swaps_with_previous_station(self):
        response = self.client.post(f"/dashboard/station/{self.station_c.pk}/move/up/")

        self.assertEqual(response.status_code, 302)
        names = list(
            Station.objects.filter(org=self.org)
            .order_by("sort_order", "id")
            .values_list("name", flat=True)
        )
        self.assertEqual(names, ["A", "C", "B"])



class RunScheduledForecastsCommandTests(TestCase):
    @patch("dashboard.management.commands.run_scheduled_forecasts.importlib.import_module")
    def test_auto_history_safe_helper_returns_zero_on_import_error(self, import_module_mock):
        import_module_mock.side_effect = RuntimeError("boom")

        class DummyStyle:
            @staticmethod
            def WARNING(value):
                return value

        class DummyStdout:
            def __init__(self):
                self.messages = []

            def write(self, message):
                self.messages.append(message)

        stdout = DummyStdout()
        result = _run_auto_history_updates_safe(stdout, DummyStyle)

        self.assertEqual(result, 0)
        self.assertTrue(any("Auto history skipped due to error" in m for m in stdout.messages))

    @patch("dashboard.management.commands.run_scheduled_forecasts.importlib.import_module")
    def test_auto_history_safe_helper_returns_rows_on_success(self, import_module_mock):
        class DummyModule:
            @staticmethod
            def run_auto_history_updates():
                return 7

        import_module_mock.return_value = DummyModule

        class DummyStyle:
            @staticmethod
            def WARNING(value):
                return value

        class DummyStdout:
            def write(self, message):
                return None

        result = _run_auto_history_updates_safe(DummyStdout(), DummyStyle)

        self.assertEqual(result, 7)

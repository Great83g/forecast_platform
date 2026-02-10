from django.contrib.auth.models import User
from django.test import TestCase
from unittest.mock import patch

from dashboard.services.forecast_engine import run_forecast_for_station
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

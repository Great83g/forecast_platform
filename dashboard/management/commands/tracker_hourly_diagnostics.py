from __future__ import annotations

from datetime import datetime

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from dashboard.services.train_models import station_capacity_mw
from dashboard.services.tracker_pvlib_training import (
    add_tracker_prediction_features,
    tracker_config_from_station,
)
from solar.models import SolarForecast, SolarRecord
from stations.models import Station


class Command(BaseCommand):
    help = "Print hourly tracker PVLIB diagnostics for one station/date range."

    def add_arguments(self, parser):
        station_group = parser.add_mutually_exclusive_group(required=True)
        station_group.add_argument("--station-id", type=int)
        station_group.add_argument("--station", dest="station_name")
        parser.add_argument("--from", dest="date_from", required=True, help="Start date, YYYY-MM-DD")
        parser.add_argument("--to", dest="date_to", required=True, help="End date, YYYY-MM-DD")
        parser.add_argument("--forecast-scope", default=SolarForecast.SCOPE_MAIN)

    def handle(self, *args, **options):
        try:
            date_from = datetime.strptime(options["date_from"], "%Y-%m-%d").date()
            date_to = datetime.strptime(options["date_to"], "%Y-%m-%d").date()
        except ValueError as exc:
            raise CommandError("Dates must be in YYYY-MM-DD format") from exc
        if date_to < date_from:
            raise CommandError("--to must be greater than or equal to --from")

        if options.get("station_id"):
            station = Station.objects.filter(pk=options["station_id"]).first()
        else:
            station = Station.objects.filter(name=options["station_name"]).first()
        if station is None:
            raise CommandError("Station not found")

        tz = timezone.get_current_timezone()
        start = timezone.datetime.combine(date_from, timezone.datetime.min.time()).replace(tzinfo=tz)
        end = timezone.datetime.combine(date_to, timezone.datetime.max.time()).replace(tzinfo=tz)

        forecasts = list(
            SolarForecast.objects.filter(
                station=station,
                forecast_scope=options["forecast_scope"],
                timestamp__gte=start,
                timestamp__lte=end,
            ).order_by("timestamp")
        )
        if not forecasts:
            self.stdout.write("No forecast rows found for the requested range/scope.")
            return

        cap_mw = station_capacity_mw(station, pd.DataFrame())
        feat = pd.DataFrame(
            {
                "ds": [f.timestamp for f in forecasts],
                "Irradiation": [f.irradiation_fc for f in forecasts],
                "air_temp": [f.air_temp_fc for f in forecasts],
                "wind_speed": [f.wind_speed_fc for f in forecasts],
            }
        )
        feat["Air_Temp"] = pd.to_numeric(feat["air_temp"], errors="coerce").fillna(0.0)
        feat["Wind_Speed"] = pd.to_numeric(feat["wind_speed"], errors="coerce").fillna(0.0)
        feat["PV_Temp"] = feat["Air_Temp"] + (
            pd.to_numeric(feat["Irradiation"], errors="coerce").fillna(0.0) / 1000.0 * 20.0
        )
        feat = add_tracker_prediction_features(feat, station, cap_mw)

        records = {
            r.timestamp: r
            for r in SolarRecord.objects.filter(
                station=station,
                history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
                timestamp__gte=start,
                timestamp__lte=end,
            )
        }
        rows = []
        for i, forecast in enumerate(forecasts):
            rec = records.get(forecast.timestamp)
            row = feat.iloc[i]
            rows.append(
                {
                    "timestamp": forecast.timestamp,
                    "hour": forecast.timestamp.hour,
                    "GHI_forecast": forecast.irradiation_fc,
                    "POA_pvlib_forecast": row.get("POA_pvlib"),
                    "POA_real_history": rec.effective_poa() if rec else None,
                    "GHI_real_history": rec.effective_ghi() if rec else None,
                    "tracker_theta": row.get("tracker_theta"),
                    "solar_elevation": row.get("solar_elevation", row.get("sun_elev_deg")),
                    "solar_azimuth": row.get("solar_azimuth"),
                    "aoi": row.get("aoi", row.get("tracker_aoi")),
                    "y_expected": row.get("tracker_pvlib_baseline_mw"),
                    "y_pred_final": (forecast.pred_final / 1000.0) if forecast.pred_final is not None else None,
                    "y_fact": (rec.power_kw / 1000.0) if rec and rec.power_kw is not None else None,
                    "error_mw": (
                        (forecast.pred_final / 1000.0) - (rec.power_kw / 1000.0)
                        if forecast.pred_final is not None and rec and rec.power_kw is not None
                        else None
                    ),
                }
            )

        cfg = tracker_config_from_station(station)
        self.stdout.write(f"station={station.name} id={station.pk} tracker_pvlib={cfg.as_meta()}")
        self.stdout.write(pd.DataFrame(rows).to_csv(index=False))

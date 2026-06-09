import logging
import sqlite3
from datetime import datetime
from typing import Any, Iterable

from solar.models import SolarForecast, SolarRecord
from stations.models import Station


logger = logging.getLogger(__name__)


def _dt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _connect_station_org_db(station: Station) -> sqlite3.Connection | None:
    org = getattr(station, "org", None)
    db_path = getattr(org, "data_db_path", "") if org is not None else ""
    if not db_path:
        return None
    return sqlite3.connect(db_path)


def _ensure_schema(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stations_station (
            id INTEGER PRIMARY KEY,
            org_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            capacity_mw REAL,
            latitude REAL,
            longitude REAL,
            timezone TEXT,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS solar_solarrecord (
            id INTEGER PRIMARY KEY,
            station_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            history_scope TEXT NOT NULL,
            irradiation REAL,
            irradiation_ghi REAL,
            irradiation_poa REAL,
            air_temp REAL,
            pv_temp REAL,
            power_kw REAL,
            FOREIGN KEY(station_id) REFERENCES stations_station(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS solar_solarforecast (
            id INTEGER PRIMARY KEY,
            station_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            forecast_scope TEXT NOT NULL,
            pred_np REAL,
            pred_xgb REAL,
            pred_heur REAL,
            pred_final REAL,
            irradiation_fc REAL,
            air_temp_fc REAL,
            wind_speed_fc REAL,
            cloudcover_fc REAL,
            humidity_fc REAL,
            precip_fc REAL,
            snowfall_fc REAL,
            snowdepth_fc REAL,
            weather_code_fc INTEGER,
            auto_snow_flag INTEGER,
            auto_fog_flag INTEGER,
            auto_winter_factor REAL,
            manual_snow_factor REAL,
            winter_factor_applied REAL,
            poa_pvlib_fc REAL,
            dni_erbs_fc REAL,
            dhi_erbs_fc REAL,
            tracker_tilt_fc REAL,
            tracker_azimuth_fc REAL,
            forecast_np_mwh REAL,
            forecast_xgb_mwh REAL,
            forecast_ensemble_base_mwh REAL,
            hist_analog_mwh REAL,
            forecast_mwh REAL,
            forecast_method TEXT,
            created_at TEXT,
            FOREIGN KEY(station_id) REFERENCES stations_station(id)
        )
        """
    )
    for col in ["irradiation_ghi", "irradiation_poa"]:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(solar_solarrecord)").fetchall()}
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE solar_solarrecord ADD COLUMN {col} REAL")

    forecast_extra_cols = {
        "poa_pvlib_fc": "REAL",
        "dni_erbs_fc": "REAL",
        "dhi_erbs_fc": "REAL",
        "tracker_tilt_fc": "REAL",
        "tracker_azimuth_fc": "REAL",
        "forecast_np_mwh": "REAL",
        "forecast_xgb_mwh": "REAL",
        "forecast_ensemble_base_mwh": "REAL",
        "hist_analog_mwh": "REAL",
        "forecast_mwh": "REAL",
        "forecast_method": "TEXT",
    }
    existing_forecast_cols = {row[1] for row in conn.execute("PRAGMA table_info(solar_solarforecast)").fetchall()}
    for col, sql_type in forecast_extra_cols.items():
        if col not in existing_forecast_cols:
            conn.execute(f"ALTER TABLE solar_solarforecast ADD COLUMN {col} {sql_type}")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_solarrecord_station_time ON solar_solarrecord(station_id, timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_solarforecast_station_time ON solar_solarforecast(station_id, timestamp)"
    )


def _upsert_station(conn: sqlite3.Connection, station: Station):
    conn.execute(
        """
        INSERT INTO stations_station(id, org_id, name, capacity_mw, latitude, longitude, timezone, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            org_id=excluded.org_id,
            name=excluded.name,
            capacity_mw=excluded.capacity_mw,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            timezone=excluded.timezone,
            updated_at=excluded.updated_at
        """,
        (
            station.id,
            station.org_id,
            station.name,
            station.capacity_mw,
            station.latitude,
            station.longitude,
            station.timezone,
            _dt(datetime.utcnow()),
        ),
    )


def sync_station(station: Station):
    conn = _connect_station_org_db(station)
    if conn is None:
        return
    try:
        _ensure_schema(conn)
        _upsert_station(conn, station)
        conn.commit()
    except Exception:
        logger.exception("Failed to sync station %s to org DB", station.id)
    finally:
        conn.close()


def sync_solar_record(record: SolarRecord):
    station = record.station
    conn = _connect_station_org_db(station)
    if conn is None:
        return
    try:
        _ensure_schema(conn)
        _upsert_station(conn, station)
        conn.execute(
            """
            INSERT INTO solar_solarrecord(id, station_id, timestamp, history_scope, irradiation, irradiation_ghi, irradiation_poa, air_temp, pv_temp, power_kw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                station_id=excluded.station_id,
                timestamp=excluded.timestamp,
                history_scope=excluded.history_scope,
                irradiation=excluded.irradiation,
                irradiation_ghi=excluded.irradiation_ghi,
                irradiation_poa=excluded.irradiation_poa,
                air_temp=excluded.air_temp,
                pv_temp=excluded.pv_temp,
                power_kw=excluded.power_kw
            """,
            (
                record.id,
                station.id,
                _dt(record.timestamp),
                record.history_scope,
                record.irradiation,
                record.irradiation_ghi,
                record.irradiation_poa,
                record.air_temp,
                record.pv_temp,
                record.power_kw,
            ),
        )
        conn.commit()
    except Exception:
        logger.exception("Failed to sync solar record %s to org DB", record.id)
    finally:
        conn.close()


def delete_solar_record(record: SolarRecord):
    conn = _connect_station_org_db(record.station)
    if conn is None:
        return
    try:
        _ensure_schema(conn)
        conn.execute("DELETE FROM solar_solarrecord WHERE id = ?", (record.id,))
        conn.commit()
    except Exception:
        logger.exception("Failed to delete solar record %s from org DB", record.id)
    finally:
        conn.close()


def sync_solar_forecast(forecast: SolarForecast):
    station = forecast.station
    conn = _connect_station_org_db(station)
    if conn is None:
        return
    try:
        _ensure_schema(conn)
        _upsert_station(conn, station)
        conn.execute(
            """
            INSERT INTO solar_solarforecast(
                id, station_id, timestamp, forecast_scope,
                pred_np, pred_xgb, pred_heur, pred_final,
                irradiation_fc, air_temp_fc, wind_speed_fc, cloudcover_fc, humidity_fc, precip_fc,
                snowfall_fc, snowdepth_fc, weather_code_fc,
                auto_snow_flag, auto_fog_flag, auto_winter_factor, manual_snow_factor, winter_factor_applied,
                poa_pvlib_fc, dni_erbs_fc, dhi_erbs_fc, tracker_tilt_fc, tracker_azimuth_fc,
                forecast_np_mwh, forecast_xgb_mwh, forecast_ensemble_base_mwh, hist_analog_mwh, forecast_mwh, forecast_method,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                station_id=excluded.station_id,
                timestamp=excluded.timestamp,
                forecast_scope=excluded.forecast_scope,
                pred_np=excluded.pred_np,
                pred_xgb=excluded.pred_xgb,
                pred_heur=excluded.pred_heur,
                pred_final=excluded.pred_final,
                irradiation_fc=excluded.irradiation_fc,
                air_temp_fc=excluded.air_temp_fc,
                wind_speed_fc=excluded.wind_speed_fc,
                cloudcover_fc=excluded.cloudcover_fc,
                humidity_fc=excluded.humidity_fc,
                precip_fc=excluded.precip_fc,
                snowfall_fc=excluded.snowfall_fc,
                snowdepth_fc=excluded.snowdepth_fc,
                weather_code_fc=excluded.weather_code_fc,
                auto_snow_flag=excluded.auto_snow_flag,
                auto_fog_flag=excluded.auto_fog_flag,
                auto_winter_factor=excluded.auto_winter_factor,
                manual_snow_factor=excluded.manual_snow_factor,
                winter_factor_applied=excluded.winter_factor_applied,
                poa_pvlib_fc=excluded.poa_pvlib_fc,
                dni_erbs_fc=excluded.dni_erbs_fc,
                dhi_erbs_fc=excluded.dhi_erbs_fc,
                tracker_tilt_fc=excluded.tracker_tilt_fc,
                tracker_azimuth_fc=excluded.tracker_azimuth_fc,
                forecast_np_mwh=excluded.forecast_np_mwh,
                forecast_xgb_mwh=excluded.forecast_xgb_mwh,
                forecast_ensemble_base_mwh=excluded.forecast_ensemble_base_mwh,
                hist_analog_mwh=excluded.hist_analog_mwh,
                forecast_mwh=excluded.forecast_mwh,
                forecast_method=excluded.forecast_method,
                created_at=excluded.created_at
            """,
            (
                forecast.id,
                station.id,
                _dt(forecast.timestamp),
                forecast.forecast_scope,
                forecast.pred_np,
                forecast.pred_xgb,
                forecast.pred_heur,
                forecast.pred_final,
                forecast.irradiation_fc,
                forecast.air_temp_fc,
                forecast.wind_speed_fc,
                forecast.cloudcover_fc,
                forecast.humidity_fc,
                forecast.precip_fc,
                forecast.snowfall_fc,
                forecast.snowdepth_fc,
                forecast.weather_code_fc,
                forecast.auto_snow_flag,
                forecast.auto_fog_flag,
                forecast.auto_winter_factor,
                forecast.manual_snow_factor,
                forecast.winter_factor_applied,
                forecast.poa_pvlib_fc,
                forecast.dni_erbs_fc,
                forecast.dhi_erbs_fc,
                forecast.tracker_tilt_fc,
                forecast.tracker_azimuth_fc,
                forecast.forecast_np_mwh,
                forecast.forecast_xgb_mwh,
                forecast.forecast_ensemble_base_mwh,
                forecast.hist_analog_mwh,
                forecast.forecast_mwh,
                forecast.forecast_method,
                _dt(forecast.created_at),
            ),
        )
        conn.commit()
    except Exception:
        logger.exception("Failed to sync solar forecast %s to org DB", forecast.id)
    finally:
        conn.close()


def delete_solar_forecast(forecast: SolarForecast):
    conn = _connect_station_org_db(forecast.station)
    if conn is None:
        return
    try:
        _ensure_schema(conn)
        conn.execute("DELETE FROM solar_solarforecast WHERE id = ?", (forecast.id,))
        conn.commit()
    except Exception:
        logger.exception("Failed to delete solar forecast %s from org DB", forecast.id)
    finally:
        conn.close()


def sync_solar_records(records: Iterable[SolarRecord]):
    for record in records:
        sync_solar_record(record)


def sync_solar_forecasts(forecasts: Iterable[SolarForecast]):
    for forecast in forecasts:
        sync_solar_forecast(forecast)

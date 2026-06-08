import sqlite3

from django.contrib.auth.models import User
from django.test import TestCase

from solar.models import SolarForecast, SolarRecord
from stations.models import Organization, Station


class OrgDatabaseMirrorTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="mirror-owner", password="pass12345")
        self.org = Organization.objects.create(name="Mirror Org", owner=self.owner)
        self.station = Station.objects.create(org=self.org, name="Mirror Station", capacity_mw=1.5)

    def test_record_and_forecast_are_written_to_org_database(self):
        record = SolarRecord.objects.create(
            station=self.station,
            timestamp="2026-01-01T10:00:00+05:00",
            irradiation=120.0,
            air_temp=5.5,
            pv_temp=9.1,
            power_kw=50.0,
        )
        forecast = SolarForecast.objects.create(
            station=self.station,
            timestamp="2026-01-01T11:00:00+05:00",
            pred_final=52.3,
            irradiation_fc=130.0,
            air_temp_fc=6.0,
        )

        conn = sqlite3.connect(self.org.data_db_path)
        cur = conn.cursor()

        cur.execute("SELECT id, name FROM stations_station WHERE id = ?", (self.station.id,))
        station_row = cur.fetchone()
        self.assertEqual(station_row, (self.station.id, self.station.name))

        cur.execute("SELECT id, station_id, power_kw FROM solar_solarrecord WHERE id = ?", (record.id,))
        rec_row = cur.fetchone()
        self.assertEqual(rec_row, (record.id, self.station.id, 50.0))

        cur.execute("SELECT id, station_id, pred_final FROM solar_solarforecast WHERE id = ?", (forecast.id,))
        fc_row = cur.fetchone()
        self.assertEqual(fc_row, (forecast.id, self.station.id, 52.3))
        conn.close()

    def test_record_delete_is_mirrored_to_org_database(self):
        record = SolarRecord.objects.create(
            station=self.station,
            timestamp="2026-01-01T10:00:00+05:00",
            power_kw=10.0,
        )
        record_id = record.id
        record.delete()

        conn = sqlite3.connect(self.org.data_db_path)
        cur = conn.cursor()
        cur.execute("SELECT id FROM solar_solarrecord WHERE id = ?", (record_id,))
        self.assertIsNone(cur.fetchone())
        conn.close()

class IrradiationSplitTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="irr-owner", password="pass12345")
        self.org = Organization.objects.create(name="Irr Org", owner=self.owner)
        self.station = Station.objects.create(org=self.org, name="Irr Station", capacity_mw=1.5)

    def test_record_effective_ghi_and_poa_preserve_legacy_type(self):
        ghi_record = SolarRecord.objects.create(
            station=self.station,
            timestamp="2026-01-01T10:00:00+05:00",
            irradiation=111.0,
            irradiation_ghi=222.0,
            irradiation_poa=333.0,
        )
        self.assertEqual(ghi_record.effective_ghi(), 222.0)
        self.assertEqual(ghi_record.effective_poa(), 333.0)

        self.station.irradiation_type = Station.IRRADIATION_POA
        self.station.save(update_fields=["irradiation_type"])
        poa_record = SolarRecord.objects.create(
            station=self.station,
            timestamp="2026-01-01T11:00:00+05:00",
            irradiation=444.0,
        )
        self.assertIsNone(poa_record.effective_ghi())
        self.assertEqual(poa_record.effective_poa(), 444.0)

    def test_org_database_mirrors_split_irradiation_columns(self):
        record = SolarRecord.objects.create(
            station=self.station,
            timestamp="2026-01-01T12:00:00+05:00",
            irradiation=100.0,
            irradiation_ghi=101.0,
            irradiation_poa=102.0,
            power_kw=10.0,
        )

        conn = sqlite3.connect(self.org.data_db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT irradiation, irradiation_ghi, irradiation_poa FROM solar_solarrecord WHERE id = ?",
            (record.id,),
        )
        self.assertEqual(cur.fetchone(), (100.0, 101.0, 102.0))
        conn.close()

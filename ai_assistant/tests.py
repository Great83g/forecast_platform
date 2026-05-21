import json
from datetime import date

from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings

from ai_assistant.date_parser import parse_period
from stations.models import Organization, Station


@override_settings(TIME_ZONE="UTC", USE_TZ=True)
class AssistantDateParserTests(TestCase):
    def test_parse_last_30_days(self):
        period = parse_period("план факт за последние 30 дней")
        self.assertEqual((period.date_to - period.date_from).days, 29)

    def test_parse_previous_week(self):
        period = parse_period("прогноз за предыдущую неделю")
        self.assertEqual(period.date_from.weekday(), 0)
        self.assertEqual(period.date_to.weekday(), 6)
        self.assertEqual((period.date_to - period.date_from).days, 6)

    def test_parse_previous_quarter(self):
        period = parse_period("план факт за предыдущий квартал")
        self.assertEqual(period.date_from.day, 1)
        self.assertEqual(period.date_from.month % 3, 1)
        self.assertGreaterEqual((period.date_to - period.date_from).days, 89)


class AssistantApiContextTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="assistant_user", password="secret123")
        self.org = Organization.objects.create(name="Org", owner=self.user)
        self.station1 = Station.objects.create(org=self.org, name="SES 1.2 MW")
        self.station2 = Station.objects.create(org=self.org, name="SES Balkhash 50 MW")
        self.client.force_login(self.user)

    def test_requires_clarification_with_multiple_stations(self):
        response = self.client.post(
            "/api/assistant/query/",
            data={"text": "план факт за текущий месяц"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertIn("choices", payload)
        self.assertGreaterEqual(len(payload["choices"]), 2)

    def test_uses_context_station_when_text_has_no_station(self):
        response = self.client.post(
            "/api/assistant/query/",
            data={
                "text": "план факт за текущий месяц",
                "context": {"station_id": self.station1.pk},
            },
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("context", {}).get("station_id"), self.station1.pk)
        self.assertIn("SES 1.2", payload.get("answer", ""))


    def test_rejects_invalid_json_payload(self):
        response = self.client.post(
            "/api/assistant/query/",
            data="{bad",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "invalid_json")
        self.assertEqual(payload.get("api_version"), "v1")

    def test_requires_text(self):
        response = self.client.post(
            "/api/assistant/query/",
            data=json.dumps({"text": "   "}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "empty_text")
        self.assertEqual(payload.get("api_version"), "v1")

    def test_unsupported_intent_returns_error_code(self):
        response = self.client.post(
            "/api/assistant/query/",
            data=json.dumps({"text": "привет ассистент"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "unsupported_intent")
        self.assertEqual(payload.get("api_version"), "v1")

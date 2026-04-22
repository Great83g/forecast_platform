from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from .services.calculator_engine import calculate


class CalculatorEngineTests(TestCase):
    def test_consumption_mode_returns_variants(self):
        payload = calculate("consumption", {"monthly_kwh": 900, "tariff": 35, "specific_yield": 1450})
        self.assertIn("result", payload)
        self.assertEqual(payload["result"]["panel_model"], "SP-N16/144HG")
        self.assertEqual(len(payload["variants"]), 3)


class CalculatorApiTests(TestCase):
    def setUp(self):
        self.client = Client()
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="calc_user", password="secret123")

    def test_calculate_api(self):
        self.client.login(username="calc_user", password="secret123")
        response = self.client.post(
            reverse("solar_calculator:calculate"),
            data={"mode": "budget", "inputs": {"budget": 5000000, "specific_yield": 1450}},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("result", data)

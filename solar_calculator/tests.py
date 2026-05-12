import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from .services.calculator_engine import calculate


class BitrixResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class CalculatorEngineTests(TestCase):
    def test_unified_response_shape(self):
        payload = calculate("consumption", {"monthly_kwh": 350})
        self.assertEqual(payload["mode"], "consumption")
        self.assertIn("result_type", payload)
        self.assertIn("meta", payload)
        self.assertIn("inputs_echo", payload)
        self.assertIn("result", payload)
        self.assertIn("variants", payload)
        self.assertIn("warnings", payload)
        self.assertIn("errors", payload)

    def test_consumption_roof_fit(self):
        payload = calculate("consumption", {"monthly_kwh": 350, "roof_area_m2": 10})
        self.assertIn(payload["result"]["roof_fit"], {"fits", "partial", "not_fit"})

    def test_consumption_export_model_enabled(self):
        payload = calculate(
            "consumption",
            {
                "monthly_kwh": 350,
                "export_enabled": True,
                "export_tariff_kzt_per_kwh": 20,
            },
        )
        self.assertEqual(payload["errors"], [])
        export_model = payload["result"]["export_model"]
        self.assertTrue(export_model["export_enabled"])
        self.assertEqual(export_model["export_tariff_kzt_per_kwh"], 20.0)
        self.assertIn("total_benefit_kzt", export_model)

    def test_roof_area_export_model_enabled(self):
        payload = calculate(
            "roof_area",
            {
                "roof_area_m2": 40,
                "roof_type": "simple",
                "monthly_kwh": 350,
                "export_enabled": True,
                "export_tariff_kzt_per_kwh": 20,
            },
        )
        self.assertEqual(payload["errors"], [])
        export_model = payload["result"]["export_model"]
        self.assertTrue(export_model["export_enabled"])
        self.assertEqual(export_model["export_tariff_kzt_per_kwh"], 20.0)

    def test_consumption_cost_breakdown_uses_residential_percentages(self):
        payload = calculate("consumption", {"monthly_kwh": 350})
        self.assertEqual(payload["errors"], [])
        assert_cost_breakdown(self, payload["result"], RESIDENTIAL_COST_BREAKDOWN_PERCENTAGES)
        self.assertEqual(
            payload["result"]["cost_breakdown"]["total_cost_kzt"],
            payload["result"]["estimated_cost_kzt"],
        )

        for variant in payload["variants"]:
            assert_cost_breakdown(self, variant, RESIDENTIAL_COST_BREAKDOWN_PERCENTAGES)
            self.assertEqual(
                variant["cost_breakdown"]["total_cost_kzt"],
                variant["estimated_cost_kzt"],
            )

    def test_roof_area_cost_breakdown_uses_residential_percentages(self):
        payload = calculate(
            "roof_area",
            {"roof_area_m2": 40, "roof_type": "simple", "monthly_kwh": 350},
        )
        self.assertEqual(payload["errors"], [])
        assert_cost_breakdown(self, payload["result"], RESIDENTIAL_COST_BREAKDOWN_PERCENTAGES)
        self.assertEqual(
            payload["result"]["cost_breakdown"]["total_cost_kzt"],
            payload["result"]["estimated_cost_kzt"],
        )

        for variant in payload["variants"]:
            assert_cost_breakdown(self, variant, RESIDENTIAL_COST_BREAKDOWN_PERCENTAGES)
            self.assertEqual(
                variant["cost_breakdown"]["total_cost_kzt"],
                variant["estimated_cost_kzt"],
            )

    def test_appliances_real_mode(self):
        payload = calculate(
            "appliances",
            {
                "appliances": [
                    {"name": "ac", "power_kw": 1.2, "hours_per_day": 6, "quantity": 2},
                    {"name": "fridge", "power_kw": 0.15, "hours_per_day": 24, "quantity": 1},
                ]
            },
        )
        self.assertEqual(payload["errors"], [])
        self.assertGreater(payload["result"]["daily_kwh"], 0)
        self.assertGreater(payload["result"]["peak_kw"], 0)

    def test_grid_export_mode(self):
        payload = calculate(
            "grid_export",
            {
                "target_kw": 500,
                "specific_yield": 1450,
                "tariff_kzt_per_kwh": 35,
                "export_tariff_kzt_per_kwh": 25,
                "own_consumption_percent": 10,
                "land_area_ha": 5,
            },
        )
        self.assertEqual(payload["result_type"], "commercial")
        self.assertEqual(payload["errors"], [])
        self.assertIn("cost_breakdown", payload)
        self.assertIn("energy_model", payload)
        self.assertEqual(payload["recommended_variant"], "grid_export")
        self.assertIn("export_revenue_kzt", payload["result"])
        self.assertIn("total_benefit_kzt", payload["result"])
        self.assertEqual(payload["result"]["price_per_kw_kzt"], 350_000.0)
        self.assertEqual(payload["result"]["estimated_cost_kzt"], 175_000_000.0)
        self.assertEqual(
            payload["result"]["cost_breakdown"]["total_cost_kzt"],
            175_000_000.0,
        )
        self.assertEqual(payload["result"]["cost_breakdown"]["panels_cost_kzt"], 61_250_000.0)

    def test_utility_power_has_variant_result_and_utility_cost_breakdown(self):
        payload = calculate(
            "utility_power",
            {"target_mw_ac": 1.2, "specific_yield": 1450, "tariff_kzt_per_kwh": 35},
        )
        expected_breakdown = {
            "panels_cost_kzt": 147_000_000.0,
            "equipment_cost_kzt": 42_000_000.0,
            "mounting_structure_cost_kzt": 67_200_000.0,
            "cables_cost_kzt": 46_200_000.0,
            "communication_system_cost_kzt": 29_400_000.0,
            "installation_commissioning_cost_kzt": 88_200_000.0,
            "total_cost_kzt": 420_000_000.0,
        }

        self.assertEqual(payload["errors"], [])
        self.assertTrue(payload["variants"])
        self.assertEqual(payload["recommended_variant"], "utility_power")
        self.assertEqual(payload["variants"][0]["name"], "utility_power")
        self.assertEqual(payload["result"], payload["variants"][0])
        self.assertEqual(
            payload["result"]["price_per_kw_kzt"],
            {"min": 350_000.0, "max": 350_000.0, "mid": 350_000.0},
        )
        self.assertEqual(payload["result"]["estimated_cost_kzt"], 420_000_000.0)
        self.assertEqual(payload["result"]["cost_breakdown"], expected_breakdown)

    def test_utility_land_has_variant_and_result_matches(self):
        payload = calculate(
            "utility_land",
            {"land_hectares": 2, "specific_yield": 1450, "tariff_kzt_per_kwh": 35},
        )
        self.assertEqual(payload["errors"], [])
        self.assertTrue(payload["variants"])
        self.assertEqual(payload["recommended_variant"], "utility_land")
        self.assertEqual(payload["variants"][0]["name"], "utility_land")
        self.assertEqual(payload["result"], payload["variants"][0])
        self.assertEqual(
            payload["result"]["price_per_kw_kzt"],
            {"min": 350_000.0, "max": 350_000.0, "mid": 350_000.0},
        )
        self.assertEqual(payload["result"]["estimated_cost_kzt"], 466_666_666.67)
        self.assertEqual(payload["result"]["cost_breakdown"]["total_cost_kzt"], 466_666_666.67)


class CalculatorApiTests(TestCase):
    def setUp(self):
        self.client = Client()
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="calc_user", password="secret123")

    def test_calculate_api(self):
        self.client.login(username="calc_user", password="secret123")
        response = self.client.post(
            reverse("solar_calculator:calculate"),
            data={"mode": "budget", "inputs": {"budget_kzt": 5_000_000}},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("result", data)
        self.assertEqual(data["errors"], [])

    def test_calculate_api_anonymous_allowed(self):
        response = self.client.post(
            reverse("solar_calculator:calculate"),
            data={"mode": "consumption", "inputs": {"monthly_kwh": 350}},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("result", data)

    def test_calculator_embed_page_is_iframe_friendly(self):
        response = self.client.get(reverse("solar_calculator:embed"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<meta name="robots" content="noindex,nofollow">')
        self.assertContains(response, 'id="modeTabs"')
        self.assertContains(response, reverse("solar_calculator:calculate"))
        self.assertNotIn("X-Frame-Options", response.headers)

    def test_calculator_page_renders_lead_url(self):
        response = self.client.get(reverse("solar_calculator:page"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, reverse("solar_calculator:lead"))

    @override_settings(BITRIX_WEBHOOK_URL="https://portal.example/rest/1/token/")
    def test_calculate_lead_api(self):
        with patch(
            "solar_calculator.views.urlopen",
            return_value=BitrixResponse({"result": 123}),
        ) as mocked_urlopen:
            response = self.client.post(
                reverse("solar_calculator:lead"),
                data={
                    "name": "Alex",
                    "phone": "+77000000000",
                    "email": "alex@example.com",
                    "comment": "хочу КП",
                    "selected_plan": "Оптимальный",
                    "price": "580 000 ₸",
                    "panel_count": "4",
                    "system_power_kw": "2.32 кВт",
                    "payback_years": "5.1 лет",
                },
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["lead_id"], 123)

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://portal.example/rest/1/token/crm.lead.add.json",
        )
        bitrix_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            bitrix_payload["fields"]["TITLE"],
            "Заявка с калькулятора СЭС — Alex",
        )
        self.assertEqual(
            bitrix_payload["fields"]["PHONE"],
            [{"VALUE": "+77000000000", "VALUE_TYPE": "WORK"}],
        )
        self.assertEqual(
            bitrix_payload["fields"]["EMAIL"],
            [{"VALUE": "alex@example.com", "VALUE_TYPE": "WORK"}],
        )
        self.assertIn("Выбранный план: Оптимальный", bitrix_payload["fields"]["COMMENTS"])
        self.assertIn("Цена: 580 000 ₸", bitrix_payload["fields"]["COMMENTS"])

    def test_calculate_lead_api_requires_name_and_phone(self):
        response = self.client.post(
            reverse("solar_calculator:lead"),
            data={"name": "", "phone": ""},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])

    def test_calculate_lead_api_reports_bitrix_error(self):
        with patch(
            "solar_calculator.views.urlopen",
            return_value=BitrixResponse({
                "error": "ERROR",
                "error_description": "Bitrix failed",
            }),
        ):
            response = self.client.post(
                reverse("solar_calculator:lead"),
                data={"name": "Alex", "phone": "+77000000000"},
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 502)
        self.assertFalse(response.json()["success"])
        self.assertEqual(response.json()["error"], "Bitrix failed")

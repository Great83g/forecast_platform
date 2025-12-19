from django.test import TestCase
from django.urls import reverse, resolve
from django.contrib.auth import views as auth_views


class LoginPageTests(TestCase):
    def test_login_page_renders(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "accounts/login.html")

    def test_login_url_resolves_to_builtin_view(self):
        match = resolve("/login/")
        self.assertEqual(match.func.view_class, auth_views.LoginView)

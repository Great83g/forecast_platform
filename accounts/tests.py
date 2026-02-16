from django.contrib.auth import get_user_model
from django.contrib.auth import views as auth_views
from django.test import TestCase
from django.urls import resolve, reverse


class LoginPageTests(TestCase):
    def test_login_page_renders(self):
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "accounts/login.html")

    def test_login_url_resolves_to_builtin_view(self):
        match = resolve("/login/")
        self.assertEqual(match.func.view_class, auth_views.LoginView)


class RegisterPageTests(TestCase):
    def test_register_page_renders(self):
        response = self.client.get(reverse("register"))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "accounts/register.html")

    def test_register_creates_user_and_redirects_to_login(self):
        response = self.client.post(
            reverse("register"),
            {
                "username": "newuser",
                "password1": "StrongPass123!",
                "password2": "StrongPass123!",
            },
        )

        self.assertRedirects(response, reverse("login"))
        self.assertTrue(get_user_model().objects.filter(username="newuser").exists())

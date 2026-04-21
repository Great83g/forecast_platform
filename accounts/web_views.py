from django.conf import settings
from django.contrib import messages
from django.contrib.auth.views import LoginView
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views.generic.edit import FormView


class LoginPageView(LoginView):
    template_name = "accounts/login.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["login_hero_video_url"] = settings.LOGIN_HERO_VIDEO_URL
        context["login_hero_poster_url"] = settings.LOGIN_HERO_POSTER_URL
        return context


class RegisterPageView(FormView):
    template_name = "accounts/register.html"
    form_class = UserCreationForm
    success_url = reverse_lazy("login")

    def form_valid(self, form):
        form.save()
        messages.success(self.request, "Регистрация успешна. Теперь войдите в систему.")
        return super().form_valid(form)

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from stations.models import Organization, OrganizationMember, Station

from .forms import WindStationForm, WindStationProfileForm


def _wind_station_queryset_for_user(user):
    org_ids = Organization.objects.filter(owner=user).values_list("id", flat=True)
    member_org_ids = OrganizationMember.objects.filter(user=user).values_list("organization_id", flat=True)
    return Station.objects.filter(org_id__in=(org_ids.union(member_org_ids)), station_kind=Station.KIND_WIND).distinct()


@login_required
def station_list(request):
    stations = list(_wind_station_queryset_for_user(request.user).select_related("org", "wind_profile").order_by("sort_order", "id"))
    return render(request, "wind/station_list.html", {"stations": stations})


@login_required
def station_create(request):
    if request.method == "POST":
        station_form = WindStationForm(request.POST, user=request.user)
        profile_form = WindStationProfileForm(request.POST)
        if station_form.is_valid() and profile_form.is_valid():
            st = station_form.save(commit=False)
            st.station_kind = Station.KIND_WIND

            profile = profile_form.save(commit=False)
            installed_capacity_kw = profile.installed_capacity_kw
            st.capacity_ac_kw = installed_capacity_kw
            st.capacity_dc_kw = installed_capacity_kw
            st.capacity_mw = installed_capacity_kw / 1000.0

            if st.data_shift_hours in (None, ""):
                st.data_shift_hours = 0

            st.save()
            profile.station = st
            profile.save()

            messages.success(request, "Ветростанция создана.")
            return redirect("wind:station-list")
        messages.error(request, "Проверьте форму: есть ошибки в параметрах ветростанции.")
    else:
        station_form = WindStationForm(user=request.user)
        profile_form = WindStationProfileForm()

    return render(
        request,
        "wind/station_create.html",
        {
            "station_form": station_form,
            "profile_form": profile_form,
        },
    )

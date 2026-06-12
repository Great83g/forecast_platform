from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Exists, OuterRef
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from stations.models import Organization, OrganizationMember, Station

from .forms import ESSSimulationRunForm, StationBalanceConfigForm, VirtualESSConfigForm
from .models import ESSSimulationRun, StationBalanceConfig, VirtualESSConfig
from .services import build_run_summary, populate_simulation_points


def _station_queryset_for_user(user):
    org_ids = Organization.objects.filter(owner=user).values_list("id", flat=True)
    member_org_ids = OrganizationMember.objects.filter(user=user).values_list("organization_id", flat=True)
    return Station.objects.filter(org_id__in=org_ids.union(member_org_ids)).distinct()


def _get_station_or_404(user, station_id: int):
    return get_object_or_404(_station_queryset_for_user(user).select_related("org"), pk=station_id)


def _get_run_or_404(user, run_id: int):
    return get_object_or_404(
        ESSSimulationRun.objects.select_related("station", "ess_config", "station__org").filter(
            station__in=_station_queryset_for_user(user)
        ),
        pk=run_id,
    )


def _get_balance_config(station: Station) -> StationBalanceConfig:
    config = StationBalanceConfig.objects.filter(station=station).order_by("id").first()
    if config:
        return config
    return StationBalanceConfig(station=station, station_capacity_mw=station.capacity_mw, grid_limit_mw=station.capacity_mw)


def _get_ess_config(station: Station) -> VirtualESSConfig:
    config = VirtualESSConfig.objects.filter(station=station).order_by("id").first()
    if config:
        return config
    return VirtualESSConfig(station=station, name=f"{station.name} Virtual ESS")


@login_required
def station_list(request):
    stations = list(
        _station_queryset_for_user(request.user)
        .select_related("org")
        .annotate(
            has_balance_config=Exists(StationBalanceConfig.objects.filter(station=OuterRef("pk"))),
            has_ess_config=Exists(VirtualESSConfig.objects.filter(station=OuterRef("pk"))),
            balance_enabled=Exists(StationBalanceConfig.objects.filter(station=OuterRef("pk"), enabled=True)),
        )
        .order_by("station_kind", "sort_order", "id")
    )
    return render(request, "virtual_ess/station_list.html", {"stations": stations})


@login_required
def station_settings(request, station_id: int):
    station = _get_station_or_404(request.user, station_id)
    balance_config = _get_balance_config(station)
    ess_config = _get_ess_config(station)

    if request.method == "POST":
        balance_form = StationBalanceConfigForm(request.POST, instance=balance_config, prefix="balance")
        ess_form = VirtualESSConfigForm(request.POST, instance=ess_config, prefix="ess")
        if balance_form.is_valid() and ess_form.is_valid():
            balance = balance_form.save(commit=False)
            balance.station = station
            balance.save()

            ess = ess_form.save(commit=False)
            ess.station = station
            ess.save()

            messages.success(request, "Настройки Virtual ESS сохранены.")
            return redirect("virtual_ess:station-list")
    else:
        balance_form = StationBalanceConfigForm(instance=balance_config, prefix="balance")
        ess_form = VirtualESSConfigForm(instance=ess_config, prefix="ess")

    return render(
        request,
        "virtual_ess/station_settings.html",
        {
            "station": station,
            "balance_form": balance_form,
            "ess_form": ess_form,
        },
    )


@login_required
def station_simulate(request, station_id: int):
    station = _get_station_or_404(request.user, station_id)
    ess_config = VirtualESSConfig.objects.filter(station=station, enabled=True).order_by("id").first()
    if ess_config is None:
        ess_config = VirtualESSConfig.objects.filter(station=station).order_by("id").first()

    today = timezone.localdate()
    initial = {"date_from": today - timedelta(days=1), "date_to": today, "simulation_type": ESSSimulationRun.TYPE_POSTFACTUM}

    if request.method == "POST":
        form = ESSSimulationRunForm(request.POST)
        if form.is_valid():
            run = form.save(commit=False)
            run.station = station
            run.ess_config = ess_config
            run.status = ESSSimulationRun.STATUS_CREATED
            run.save()
            stats = populate_simulation_points(run)
            messages.success(
                request,
                "Данные симуляции загружены: "
                f"прогноз={stats['forecast_rows']}, факт={stats['actual_rows']}, точки={stats['points']}.",
            )
            return redirect("virtual_ess:run-detail", run_id=run.pk)
    else:
        form = ESSSimulationRunForm(initial=initial)

    recent_runs = ESSSimulationRun.objects.filter(station=station).select_related("ess_config").order_by("-created_at")[:10]
    return render(
        request,
        "virtual_ess/station_simulate.html",
        {
            "station": station,
            "ess_config": ess_config,
            "form": form,
            "recent_runs": recent_runs,
        },
    )


def _point_float(point, field_name: str):
    value = getattr(point, field_name)
    return float(value) if value is not None else None


def _build_run_chart_data(points) -> dict:
    point_list = list(points)
    return {
        "labels": [timezone.localtime(point.timestamp).strftime("%d.%m %H:%M") for point in point_list],
        "plan": [_point_float(point, "plan_mw") for point in point_list],
        "fact": [_point_float(point, "fact_mw") for point in point_list],
        "afterEss": [_point_float(point, "output_after_ess_mw") for point in point_list],
        "discharge": [_point_float(point, "ess_discharge_mw") for point in point_list],
        "charge": [_point_float(point, "ess_charge_mw") for point in point_list],
        "unbalanced": [_point_float(point, "unbalanced_mw") for point in point_list],
        "soc": [_point_float(point, "soc_percent") for point in point_list],
    }


@login_required
def run_detail(request, run_id: int):
    run = _get_run_or_404(request.user, run_id)
    points = list(run.points.order_by("timestamp", "id"))
    summary = build_run_summary(run)
    chart_data = _build_run_chart_data(points)
    return render(
        request,
        "virtual_ess/run_detail.html",
        {
            "run": run,
            "station": run.station,
            "points": points,
            "summary": summary,
            "chart_data": chart_data,
        },
    )

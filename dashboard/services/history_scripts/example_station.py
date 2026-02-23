import pandas as pd
from django.utils import timezone


def build_history_dataframe(station):
    now = timezone.now().replace(minute=0, second=0, microsecond=0)
    return pd.DataFrame(
        [
            {
                "ds": pd.Timestamp(now),
                "irradiation": 500.1,
                "air_temp": 20.2,
                "pv_temp": 24.3,
                "power_kw": 700.4,
            }
        ]
    )

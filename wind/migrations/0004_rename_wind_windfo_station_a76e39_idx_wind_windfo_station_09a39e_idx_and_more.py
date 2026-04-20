from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("wind", "0003_windforecast"),
    ]

    operations = [
        migrations.RenameIndex(
            model_name="windforecast",
            old_name="wind_windfo_station_a76e39_idx",
            new_name="wind_windfo_station_09a39e_idx",
        ),
        migrations.RenameIndex(
            model_name="windforecast",
            old_name="wind_windfo_station_1912bf_idx",
            new_name="wind_windfo_station_cea399_idx",
        ),
        migrations.RenameIndex(
            model_name="windrecord",
            old_name="wind_windre_station_bf6cd4_idx",
            new_name="wind_windre_station_a3073c_idx",
        ),
        migrations.RenameIndex(
            model_name="windrecord",
            old_name="wind_windre_station_a47ce9_idx",
            new_name="wind_windre_station_9dbbbe_idx",
        ),
    ]


#!/bin/bash



poetry run python dan_weather_suite/ai_models/gfs_pangu.py
poetry run ai-models --file gfsinput.grib --lead-time 168 --num-thread 8 panguweather
rm images/pangu*
poetry run python dan_weather_suite/ai_models/panguweather.py

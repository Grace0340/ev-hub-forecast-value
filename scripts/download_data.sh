#!/usr/bin/env bash
# Download the public input data into data/. No data are distributed with this repository.
#   UrbanEV station-level archive (Dryad), CHARGED Shenzhen / Los Angeles / Melbourne (GitHub),
#   NASA POWER hourly irradiance and 2-m temperature at one point per city.
set -u
ROOT=${EV_HUB_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

# UrbanEV (Li et al., Scientific Data 2025): full archive with the station-level 5-min files
D=$ROOT/data/dryad
mkdir -p "$D"
for id in 4587831:README.md 4587805:UrbanEVDataset.zip; do
  fid=${id%%:*}; name=${id#*:}
  if [ ! -s "$D/$name" ]; then
    curl -sL -m 3600 --retry 3 -A "$UA" -o "$D/$name" "https://datadryad.org/downloads/file_stream/$fid" ||
      curl -sL -m 3600 --retry 3 -A "$UA" -o "$D/$name" "https://datadryad.org/stash/downloads/file_stream/$fid"
  fi
  echo "UrbanEV $name"
done
if [ ! -d "$D/UrbanEVDataset/20220901-20230228_station-processed/features" ]; then
  unzip -q -n "$D/UrbanEVDataset.zip" "UrbanEVDataset/20220901-20230228_station-processed/features/*" -d "$D"
fi

# CHARGED (https://github.com/IntelligentSystemsLab/CHARGED): *_remove_zero city folders
BASE=https://raw.githubusercontent.com/IntelligentSystemsLab/CHARGED/main/data
for c in SZH LOA MEL; do
  mkdir -p "$ROOT/data/charged/$c"
  for f in sites.csv volume.csv; do
    dst=$ROOT/data/charged/$c/$f
    [ -s "$dst" ] || curl -sL -m 1800 --retry 3 -o "$dst" "$BASE/${c}_remove_zero/$f"
    echo "CHARGED $c $f"
  done
done

# NASA POWER (https://power.larc.nasa.gov), hourly, local standard time
N=$ROOT/data/nasa
mkdir -p "$N"
API="https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&format=CSV&time-standard=LST"
while read -r name lat lon start end; do
  [ -s "$N/$name" ] || curl -sL -m 600 --retry 3 -o "$N/$name" \
    "$API&latitude=$lat&longitude=$lon&start=$start&end=$end"
  echo "NASA POWER $name"
done <<'EOF'
shenzhen_hourly.csv 22.6 114.06 20220825 20230305
szh_2023_hourly.csv 22.6 114.06 20230320 20231005
loa_2023_hourly.csv 34.05 -118.25 20230320 20231005
mel_2023_hourly.csv -37.81 144.96 20230320 20231005
EOF

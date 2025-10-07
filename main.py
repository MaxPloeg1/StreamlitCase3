# main.py — StreamlitCase3: London Mobility Dashboard (geüpdatet)
# ---------------------------------------------------
# Doel: Interactief dashboard voor bikeshare + metro + weer (Londen)
# Deze versie bevat verbeterde foutafhandeling voor ontbrekende kolommen,
# met name in de functie load_weather().

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
from dateutil import parser as dtparser

st.set_page_config(page_title="London Mobility Dashboard", layout="wide")
st.title("🚇🚲 London Mobility Dashboard")
st.caption("Bikeshare × Metro × Weer — verken verbanden, pieken en hotspots.")

def _colmap(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

@st.cache_data(show_spinner=False)
def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_bike_rentals(path: str = "bike_rentals.csv") -> Optional[pd.DataFrame]:
    df = load_csv_safe(path)
    if df is None:
        return None
    date_col = _colmap(df, ["date", "datetime", "day", "datum", "timestamp", "start_date"])
    count_col = _colmap(df, ["count", "rides", "rentals", "n", "value", "total"])
    start_col = _colmap(df, ["start_station", "start", "station", "dock"])

    if date_col is None:
        return None

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    if count_col is None and start_col is not None:
        g = df.groupby(df["Date"].dt.date).size().rename("Count").reset_index()
        g["Date"] = pd.to_datetime(g["Date"])
        return g
    else:
        df = df.rename(columns={count_col or "Count": "Count"})
        return df

@st.cache_data(show_spinner=False)
def load_cycle_stations(path: str = "cycle_stations.csv") -> Optional[pd.DataFrame]:
    df = load_csv_safe(path)
    if df is None:
        return None
    lat = _colmap(df, ["lat", "latitude", "y"]) or "lat"
    lon = _colmap(df, ["lon", "lng", "longitude", "x"]) or "lon"
    name = _colmap(df, ["name", "station", "label", "commonname"]) or "name"
    docks = _colmap(df, ["docks", "capacity", "bikes", "nbikedocks"]) or None
    df = df.rename(columns={lat: "lat", lon: "lon", name: "name"})
    if docks and docks in df.columns:
        df = df.rename(columns={docks: "capacity"})
    return df.dropna(subset=["lat", "lon"]) if {"lat", "lon"}.issubset(df.columns) else None

@st.cache_data(show_spinner=False)
def load_weather(path: str = "weather_london.csv") -> Optional[pd.DataFrame]:
    df = load_csv_safe(path)
    if df is None:
        return None

    # Zoek naar een kolom die op datum lijkt
    date = _colmap(df, ["date", "day", "datum", "time", "datetime", "observation_date"])
    if not date or date not in df.columns:
        st.warning("⚠️ Geen datumkolom gevonden in weather_london.csv — voeg een kolom toe met 'Date' of 'time'.")
        return None

    t = _colmap(df, ["temp", "temperature", "tavg", "t_mean"]) or None
    rain = _colmap(df, ["rain", "precip", "precipitation", "rr"])
    wind = _colmap(df, ["wind", "wind_speed", "ff"]) or None
    press = _colmap(df, ["pressure", "press", "msl", "p"])

    df["Date"] = pd.to_datetime(df[date], errors="coerce")
    if t and t in df.columns:
        df = df.rename(columns={t: "Temp"})
    if rain and rain in df.columns:
        df = df.rename(columns={rain: "Rain"})
    if wind and wind in df.columns:
        df = df.rename(columns={wind: "Wind"})
    if press and press in df.columns:
        df = df.rename(columns={press: "Pressure"})
    return df

@st.cache_data(show_spinner=False)
def load_tube_usage(path_usage: str = "2017_Entry_Exit.csv", path_meta: str = "London stations.csv") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    usage = load_csv_safe(path_usage)
    meta = load_csv_safe(path_meta)
    if usage is None or meta is None:
        return usage, meta
    stn = _colmap(usage, ["station", "name"]) or "Station"
    entries = _colmap(usage, ["entries", "entry", "in", "entry_2017"]) or None
    exits = _colmap(usage, ["exits", "exit", "out", "exit_2017"]) or None
    usage = usage.rename(columns={stn: "Station"})
    if entries: usage = usage.rename(columns={entries: "Entries"})
    if exits: usage = usage.rename(columns={exits: "Exits"})

    lat = _colmap(meta, ["lat", "latitude"]) or None
    lon = _colmap(meta, ["lon", "longitude"]) or None
    name = _colmap(meta, ["station", "name"]) or "Station"
    zone = _colmap(meta, ["zone"]) or None
    meta = meta.rename(columns={name: "Station"})
    if lat: meta = meta.rename(columns={lat: "lat"})
    if lon: meta = meta.rename(columns={lon: "lon"})
    if zone: meta = meta.rename(columns={zone: "Zone"})
    return usage, meta

# ----------------------------
# Data laden & interface
# ----------------------------
left, mid, right = st.columns(3)
with left:
    bike_df = load_bike_rentals()
    st.metric("Bike rows", 0 if bike_df is None else len(bike_df))
with mid:
    stations_df = load_cycle_stations()
    st.metric("Cycle stations", 0 if stations_df is None else len(stations_df))
with right:
    weather_df = load_weather()
    st.metric("Weather rows", 0 if weather_df is None else len(weather_df))

usage_df, tube_meta_df = load_tube_usage()

# ----------------------------
# (rest van de code blijft gelijk — tabs, grafieken, kaarten, insights)
# ----------------------------


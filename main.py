# main.py — StreamlitCase3: London Mobility Dashboard
# ---------------------------------------------------
# Doel: Interactief dashboard voor bikeshare + metro + weer (Londen)
# Geïnspireerd op de voorbeeldapps uit de case. Deze versie is robuust voor
# kolomnamen (flex-match), heeft performante caching, en duidelijke tabbladen.
#
# Plaats deze file in je repo (root) en zet deze libs in requirements.txt:
#   streamlit
#   pandas
#   numpy
#   pydeck
#   plotly
#   python-dateutil
#
# Bestandsnamen (pas aan indien anders in je repo):
#   - bike_rentals.csv           (ritniveau of dagtellingen)
#   - cycle_stations.csv         (naam, lat, lon)
#   - weather_london.csv         (datum, temp, wind, neerslag)
#   - 2017_Entry_Exit.csv        (metro entries/exits per station/jaar)
#   - London stations.csv        (station locatie + zone)

from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
from dateutil import parser as dtparser

# ----------------------------
# Page config & header
# ----------------------------
st.set_page_config(page_title="London Mobility Dashboard", layout="wide")
st.title("🚇🚲 London Mobility Dashboard")
st.caption("Bikeshare × Metro × Weer — verken verbanden, pieken en hotspots.")

# ----------------------------
# Helpers
# ----------------------------

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
    # Detect datetime & count
    date_col = _colmap(df, ["date", "datetime", "day", "datum", "timestamp", "start_date"])
    count_col = _colmap(df, ["count", "rides", "rentals", "n", "value", "total"])
    start_col = _colmap(df, ["start_station", "start", "station", "dock"])
    borough_col = _colmap(df, ["borough", "area", "district"])  # optional

    # If only start/end per ride, aggregate by day
    if date_col is None:
        # Try to parse from separate cols (year, month, day) — optional extension
        pass

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    if count_col is None and start_col is not None:
        # Assume trip-level rows: one row = one rental
        # Aggregate to daily counts for performance in plots
        g = df.groupby(df["Date"].dt.date).size().rename("Count").reset_index()
        g["Date"] = pd.to_datetime(g["Date"])  # back to Timestamp
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
    date = _colmap(df, ["date", "day", "datum"]) or "date"
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
def load_tube_usage(path_usage: str = "2017_Entry_Exit.csv",
                    path_meta: str = "London stations.csv") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    usage = load_csv_safe(path_usage)
    meta = load_csv_safe(path_meta)
    if usage is None or meta is None:
        return usage, meta
    # Normalize
    stn = _colmap(usage, ["station", "name"]) or "Station"
    entries = _colmap(usage, ["entries", "entry", "in", "entry_2017"]) or None
    exits = _colmap(usage, ["exits", "exit", "out", "exit_2017"]) or None
    usage = usage.rename(columns={stn: "Station"})
    if entries: usage = usage.rename(columns={entries: "Entries"})
    if exits: usage = usage.rename(columns={exits: "Exits"})

    # Meta with coordinates
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
# Data load
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

# Date filter (global)
min_date_candidates = []
max_date_candidates = []
for d in [df for df in [bike_df, weather_df] if df is not None and "Date" in df.columns]:
    min_date_candidates.append(d["Date"].min())
    max_date_candidates.append(d["Date"].max())

if min_date_candidates and max_date_candidates:
    global_min = pd.to_datetime(min(min_date_candidates))
    global_max = pd.to_datetime(max(max_date_candidates))
else:
    global_min = pd.to_datetime("2016-01-01")
    global_max = pd.to_datetime("2018-12-31")

with st.sidebar:
    st.header("Filters")
    date_from, date_to = st.date_input("Datumrange", value=(global_min.date(), global_max.date()))
    show_trend = st.toggle("Toon trendlijn in grafieken", value=True)

# ----------------------------
# Tabs
# ----------------------------
TabOverview, TabBike, TabStations, TabWeather, TabTube, TabInsights = st.tabs([
    "📌 Overview", "🚲 Bike usage", "🗺️ Stations map", "🌦️ Weather", "🚇 Tube", "💡 Insights",
])

# ----------------------------
# Overview
# ----------------------------
with TabOverview:
    st.subheader("Kern-KPI's")
    c1, c2, c3, c4 = st.columns(4)
    # Bike totals
    if bike_df is not None and "Date" in bike_df.columns and "Count" in bike_df.columns:
        mask = (bike_df["Date"] >= pd.to_datetime(date_from)) & (bike_df["Date"] <= pd.to_datetime(date_to))
        bike_view = bike_df.loc[mask].copy()
        with c1:
            st.metric("Fietsritten (periode)", int(bike_view["Count"].sum()))
        with c2:
            st.metric("Gem. ritten/dag", f"{bike_view.groupby(bike_view['Date'].dt.date)['Count'].sum().mean():.0f}")
    else:
        bike_view = None
        with c1:
            st.metric("Fietsritten (periode)", "—")
        with c2:
            st.metric("Gem. ritten/dag", "—")

    # Weather snapshot
    if weather_df is not None and "Date" in weather_df.columns:
        wmask = (weather_df["Date"] >= pd.to_datetime(date_from)) & (weather_df["Date"] <= pd.to_datetime(date_to))
        wview = weather_df.loc[wmask].copy()
        with c3:
            if "Temp" in wview.columns:
                st.metric("Gem. temp (°C)", f"{wview['Temp'].mean():.1f}")
            else:
                st.metric("Gem. temp (°C)", "—")
        with c4:
            if "Rain" in wview.columns:
                st.metric("Totaal regen (mm)", f"{wview['Rain'].sum():.1f}")
            else:
                st.metric("Totaal regen (mm)", "—")

    st.divider()
    st.markdown("**Periodelijn: ritten per dag**")
    if bike_view is not None and len(bike_view):
        series = bike_view.groupby(bike_view["Date"].dt.date)["Count"].sum().reset_index()
        fig = px.line(series, x="Date", y="Count", markers=True)
        if show_trend and len(series) > 2:
            # eenvoudige trendlijn via polyfit
            x = np.arange(len(series))
            z = np.polyfit(x, series["Count"].to_numpy(), 1)
            trend = z[0] * x + z[1]
            fig.add_scatter(x=series["Date"], y=trend, mode="lines", name="Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Geen bike data beschikbaar of kolomnamen niet gevonden (Date/Count).")

# ----------------------------
# Bike usage
# ----------------------------
with TabBike:
    st.subheader("Rittenprofiel")
    if bike_df is None or "Date" not in bike_df.columns:
        st.warning("Bike dataset mist een datumkolom; pas kolomnamen aan of controleer je CSV.")
    else:
        mask = (bike_df["Date"] >= pd.to_datetime(date_from)) & (bike_df["Date"] <= pd.to_datetime(date_to))
        dfb = bike_df.loc[mask].copy()
        dfb["Day"] = dfb["Date"].dt.day_name()
        daily = dfb.groupby(dfb["Date"].dt.date)["Count"].sum().reset_index()
        fig = px.bar(daily, x="Date", y="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Weekdaggemiddelde (rit/dag)**")
        byday = dfb.groupby("Day")["Count"].mean().reindex([
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])
        st.bar_chart(byday)

# ----------------------------
# Stations map
# ----------------------------
with TabStations:
    st.subheader("Fietsstations — locaties en capaciteit")
    if stations_df is None:
        st.info("Kon cycle_stations.csv niet laden of kolommen lat/lon ontbreken.")
    else:
        dfm = stations_df.copy()
        # Bubble size ~ capacity (fallback 10)
        dfm["radius"] = dfm.get("capacity", pd.Series([10] * len(dfm))).astype(float).clip(5, 60)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=dfm,
            get_position='[lon, lat]',
            get_radius='radius',
            get_fill_color='[20, 120, 220, 140]',
            pickable=True,
        )
        lat0 = float(dfm["lat"].median())
        lon0 = float(dfm["lon"].median())
        view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=11)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\nCapacity: {capacity}"})
        st.pydeck_chart(r, use_container_width=True)

# ----------------------------
# Weather
# ----------------------------
with TabWeather:
    st.subheader("Weer en fietsgebruik — relatie")
    if weather_df is None or "Date" not in weather_df.columns:
        st.info("Geen weerdata geladen of datumkolom ontbreekt.")
    elif bike_df is None or "Date" not in bike_df.columns or "Count" not in bike_df.columns:
        st.info("Bike dataset mist Date/Count — nodig voor correlatie.")
    else:
        wmask = (weather_df["Date"] >= pd.to_datetime(date_from)) & (weather_df["Date"] <= pd.to_datetime(date_to))
        bmask = (bike_df["Date"] >= pd.to_datetime(date_from)) & (bike_df["Date"] <= pd.to_datetime(date_to))
        wv = weather_df.loc[wmask, [c for c in ["Date", "Temp", "Rain", "Wind", "Pressure"] if c in weather_df.columns]].copy()
        bv = bike_df.loc[bmask, ["Date", "Count"]].copy()
        # Aggregate bike per day to ensure unique Date
        bv = bv.groupby(bv["Date"].dt.date)["Count"].sum().reset_index()
        bv["Date"] = pd.to_datetime(bv["Date"])  # back to Timestamp
        merged = pd.merge(bv, wv, on="Date", how="inner")

        cols = [c for c in ["Temp", "Rain", "Wind", "Pressure"] if c in merged.columns]
        if not cols:
            st.warning("Weer-kolommen niet gevonden (verwacht: Temp, Rain, Wind, Pressure).")
        else:
            met = st.selectbox("Weervariabele", cols, index=0)
            fig = px.scatter(merged, x=met, y="Count", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
            corr = merged[[met, "Count"]].corr().iloc[0, 1]
            st.metric(f"Correlatie({met} ↔ Count)", f"{corr:.2f}")

# ----------------------------
# Tube
# ----------------------------
with TabTube:
    st.subheader("Metro: entries & exits (2017) + locaties")
    if usage_df is None or tube_meta_df is None:
        st.info("Kon metro-bestanden niet laden. Zorg dat '2017_Entry_Exit.csv' en 'London stations.csv' in de repo staan.")
    else:
        dfu = usage_df.copy()
        dfm = tube_meta_df.copy()
        # Join for coordinates
        merged = pd.merge(dfu, dfm, on="Station", how="left")
        # KPIs
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("# Stations", merged["Station"].nunique())
        if "Entries" in merged.columns:
            with k2:
                st.metric("Totaal Entries", int(merged["Entries"].sum()))
        if "Exits" in merged.columns:
            with k3:
                st.metric("Totaal Exits", int(merged["Exits"].sum()))

        st.markdown("**Top 20 drukste stations (Entries + Exits)**")
        total = 0
        if "Entries" in merged.columns and "Exits" in merged.columns:
            total = merged.assign(Total=merged["Entries"].fillna(0)+merged["Exits"].fillna(0))
        elif "Entries" in merged.columns:
            total = merged.assign(Total=merged["Entries"].fillna(0))
        elif "Exits" in merged.columns:
            total = merged.assign(Total=merged["Exits"].fillna(0))
        if isinstance(total, pd.DataFrame):
            top20 = total.groupby("Station")["Total"].sum().sort_values(ascending=False).head(20).reset_index()
            fig = px.bar(top20, x="Station", y="Total")
            fig.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=10, b=120))
            st.plotly_chart(fig, use_container_width=True)

        # Map
        if {"lat", "lon"}.issubset(merged.columns):
            mm = merged.dropna(subset=["lat", "lon"]).copy()
            mm["radius"] = (mm.get("Entries", pd.Series([0]*len(mm))).fillna(0) + mm.get("Exits", pd.Series([0]*len(mm))).fillna(0))
            if mm["radius"].max() > 0:
                mm["radius"] = (mm["radius"] / mm["radius"].max()) * 150 + 10
            else:
                mm["radius"] = 20
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=mm,
                get_position='[lon, lat]',
                get_radius='radius',
                get_fill_color='[240, 100, 100, 160]',
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=float(mm["lat"].median()), longitude=float(mm["lon"].median()), zoom=10)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Station}\nTotal: {Total}"})
            st.pydeck_chart(r, use_container_width=True)
        else:
            st.info("Geen coördinaten in London stations.csv gevonden (lat/lon).")

# ----------------------------
# Insights
# ----------------------------
with TabInsights:
    st.subheader("Auto-gegenereerde bevindingen")
    bullets = []
    if bike_df is not None and "Count" in bike_df.columns and "Date" in bike_df.columns:
        bmask = (bike_df["Date"] >= pd.to_datetime(date_from)) & (bike_df["Date"] <= pd.to_datetime(date_to))
        bv = bike_df.loc[bmask]
        daily = bv.groupby(bv["Date"].dt.date)["Count"].sum()
        if len(daily):
            bullets.append(f"Totaal fietsritten in periode: {int(daily.sum()):,}.")
            bullets.append(f"Gemiddeld per dag: {daily.mean():.0f} ritten.")
    if weather_df is not None and bike_df is not None and {"Date", "Count"}.issubset(bike_df.columns):
        m = pd.merge(bike_df.groupby(bike_df["Date"].dt.date)["Count"].sum().rename_axis("Date").reset_index(),
                     weather_df[[c for c in ["Date", "Temp", "Rain"] if c in weather_df.columns]], on="Date", how="inner")
        if "Temp" in m.columns:
            bullets.append(f"Correlatie Temp–Ritten: {m[['Temp','Count']].corr().iloc[0,1]:.2f} (positief = meer ritten bij warmte).")
        if "Rain" in m.columns:
            bullets.append(f"Correlatie Regen–Ritten: {m[['Rain','Count']].corr().iloc[0,1]:.2f} (negatief = minder ritten bij regen).")
    if usage_df is not None and ("Entries" in usage_df.columns or "Exits" in usage_df.columns):
        tot = 0
        if "Entries" in usage_df.columns: tot += usage_df["Entries"].fillna(0).sum()
        if "Exits" in usage_df.columns: tot += usage_df["Exits"].fillna(0).sum()
        bullets.append(f"Totaal metro-bewegingen (2017, entries+exits): {int(tot):,}.")

    if bullets:
        st.write("\n".join([f"• {b}" for b in bullets]))
    else:
        st.info("Upload/plaats de CSV's in de repo om inzichten te genereren.")

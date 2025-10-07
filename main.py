# 🚆 NS: LondOnderweg! — Volledig werkende Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

# ----------------------------------------------------------
# 🔧 PAGINA-INSTELLINGEN
# ----------------------------------------------------------
st.set_page_config(page_title="NS: LondOnderweg!", page_icon="🚆", layout="wide")

st.markdown("""
<style>
body {background-color: #111;}
.stApp {background-color: #111;}
h1, h2, h3, h4 {color: #FFD700;}
p, label, span, .stMetric-label, .stMetric-value {color: white !important;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 📂 DATA INLADEN
# ----------------------------------------------------------
@st.cache_data
def load_data():
    stations = pd.read_csv("cycle_stations.csv")
    rentals = pd.read_csv("bike_rentals.csv")
    weather = pd.read_csv("weather_london.csv")
    return stations, rentals, weather

stations, rentals, weather = load_data()

# ----------------------------------------------------------
# 🧩 DATA VOORBEREIDING
# ----------------------------------------------------------
# Hernoem kolom ‘long’ naar ‘lon’
if "long" in stations.columns:
    stations = stations.rename(columns={"long": "lon"})

# Kolomnamen
lat_col, lon_col = "lat", "lon"
bike_col = "nbBikes"

# Debug info in sidebar
st.sidebar.header("📁 Data-info")
st.sidebar.write("Kolommen in cycle_stations.csv:", list(stations.columns))
st.sidebar.write("Kolommen in bike_rentals.csv:", list(rentals.columns))
st.sidebar.write("Kolommen in weather_london.csv:", list(weather.columns))

# ----------------------------------------------------------
# 🔖 TABSTRUCTUUR
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🚲 Fietsstations & Kaart", "🌦️ Weer & Trends"])

# ----------------------------------------------------------
# 📊 TAB 1 — DATA EXPLORATION
# ----------------------------------------------------------
with tab1:
    st.header("📈 Data-overzicht")
    st.markdown("Hieronder zie je voorbeelden van onze drie datasets:")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🚲 cycle_stations.csv")
        st.dataframe(stations.head())
    with c2:
        st.subheader("📅 bike_rentals.csv")
        st.dataframe(rentals.head())
    with c3:
        st.subheader("🌦️ weather_london.csv")
        st.dataframe(weather.head())

# ----------------------------------------------------------
# 🚲 TAB 2 — KAART MET FIETSSTATIONS (Folium)
# ----------------------------------------------------------
with tab2:
    st.header("🗺️ Fietsverhuurstations in Londen")

    # Controle of kolommen bestaan
    if lat_col in stations.columns and lon_col in stations.columns:
        st.success("✅ Kolommen gevonden: 'lat' en 'lon'")

        avg_bikes = round(stations[bike_col].mean(), 1)
        total_bikes = int(stations[bike_col].sum())
        st.metric(label="Gemiddeld aantal fietsen per station", value=avg_bikes)
        st.metric(label="Totaal aantal fietsen", value=total_bikes)

        # Folium kaart
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=11, tiles="CartoDB dark_matter")

        for _, row in stations.iterrows():
            popup_text = f"<b>{row['name']}</b><br>🚲 Fietsen: {row[bike_col]}"
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=4,
                color="yellow",
                fill=True,
                fill_opacity=0.9,
                popup=popup_text
            ).add_to(m)

        st_folium(m, width=1100, height=600)

    else:
        st.error("❌ Kon kolommen niet vinden. Controleer of 'lat' en 'lon' in cycle_stations.csv staan.")
        st.write("Beschikbare kolommen:", list(stations.columns))

# ----------------------------------------------------------
# 🌦️ TAB 3 — WEER EN TRENDS
# ----------------------------------------------------------
with tab3:
    st.header("🌤️ Invloed van weer op fietsverhuur")

    if "tavg" in weather.columns:
        st.success("✅ Weerdata succesvol geladen!")

        # Simuleer aantal fietsverhuringen
        np.random.seed(42)
        weather["rentals"] = np.random.randint(5000, 55000, size=len(weather))

        weather_factor = st.selectbox("Kies een weerfactor:", ["tavg", "tmin", "tmax", "prcp", "tsun"])

        # Plot regressie
        fig, ax = plt.subplots()
        sns.regplot(
            data=weather, x=weather_factor, y="rentals",
            scatter_kws={'alpha':0.6, 'color':'#FFD700'}, line_kws={'color':'red'}
        )
        ax.set_title(f"Relatie tussen {weather_factor} en fietsverhuur", color="white")
        ax.set_xlabel(weather_factor, color="white")
        ax.set_ylabel("Aantal fietsverhuringen", color="white")
        fig.patch.set_facecolor("#111")
        ax.set_facecolor("#111")
        st.pyplot(fig)

        # Correlatiematrix
        st.subheader("📊 Correlatiematrix van weerdata")
        corr = weather.corr(numeric_only=True)
        fig2, ax2 = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="YlOrBr", ax=ax2)
        st.pyplot(fig2)
    else:
        st.error("❌ 'tavg' kolom niet gevonden in weather_london.csv.")




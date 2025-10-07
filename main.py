import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk

# --- CONFIG ---
st.set_page_config(page_title="NS: LondOnderweg!", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
body {
    background-color: #FFD700;
    color: #003082;
}
[data-testid="stSidebar"] {
    background-color: #003082;
    color: white;
}
h1, h2, h3, h4 {
    color: #003082;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --- TITEL ---
st.title("🚆 NS: LondOnderweg!")
st.image("https://upload.wikimedia.org/wikipedia/commons/2/29/NS_Fiets_station.jpg", use_column_width=True)
st.markdown("**Welkom bij ons mobiliteitsdashboard!** Ontdek hoe weer, metro en fietsen samen Londen in beweging houden.")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Data Exploration", "🗺️ London Maps", "📈 Predictions", "💡 Conclusions", "ℹ️ About"]
)

# --- TAB 1: Data Exploration ---
with tab1:
    st.header("Fietsverhuur en Weerdata")

    # Data inladen
    bike = pd.read_csv("bike_rentals.csv")
    weather = pd.read_csv("weather_london.csv")

    st.subheader("Debug: controleer kolomnamen")
    stations = pd.read_csv("cycle_stations.csv")
    st.write("Kolomnamen gevonden in dataset:", list(stations.columns))
    st.dataframe(stations.head())

    # Toon kolomnamen (handig voor debuggen)
    st.write("📋 Kolomnamen in cycle_stations.csv:", list(stations.columns))
    st.write("📋 Kolommen in bike_rentals.csv:", list(bike.columns))
    st.write("📋 Kolommen in weather_london.csv:", list(weather.columns))

    # Slimme automatische herkenning van relevante kolommen
    bike_col = None
    for c in bike.columns:
        if any(x in c.lower() for x in ["count", "rentals", "number", "total", "rides"]):
            bike_col = c
            break

    temp_col = None
    for c in weather.columns:
        if any(x in c.lower() for x in ["tavg", "temp", "temperature", "mean"]):
            temp_col = c
            break

    # Controle
    if not bike_col or not temp_col:
        st.error(f"❌ Kon kolommen niet vinden. Gevonden fiets={bike_col}, weer={temp_col}")
    else:
        st.success(f"Gevonden kolommen → Fiets: `{bike_col}` | Weer: `{temp_col}`")

        merged = pd.concat([bike[bike_col], weather[temp_col]], axis=1)
        merged.columns = ["BikeRentals", "Temperature"]

        fig = px.scatter(
            merged, x="Temperature", y="BikeRentals",
            trendline="ols", title="Relatie tussen temperatuur en fietshuur"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: London Maps ---
with tab2:
    st.header("London Fiets- en Metrokaarten")

    stations = pd.read_csv("cycle_stations.csv")
    st.write("📋 Kolommen in cycle_stations.csv:", list(stations.columns))

    # Automatische herkenning van latitude/longitude
    lat_col = next((c for c in stations.columns if any(x in c.lower() for x in ["lat", "latitude", "y"])), None)
    lon_col = next((c for c in stations.columns if any(x in c.lower() for x in ["lon", "lng", "longitude", "x"])), None)

    if lat_col and lon_col:
        st.success(f"Gevonden coördinatenkolommen → lat: {lat_col}, lon: {lon_col}")
        st.map(stations[[lat_col, lon_col]].dropna(), zoom=11)
    else:
        st.error(f"❌ Geen coördinatenkolommen gevonden. Gevonden lat={lat_col}, lon={lon_col}")
        st.info("Controleer of je kolommen heten zoals 'Latitude' en 'Longitude'.")
        
# --- TAB 3: Predictions ---
with tab3:
    st.header("Voorspelling Fietsverhuur")
    st.markdown("Gebruik een eenvoudige regressie om het aantal verhuurde fietsen te schatten bij bepaalde temperaturen.")
    temp = st.slider("Temperatuur (°C)", -5, 25, 10)
    y_pred = 1338.38 * temp + 15929.54
    st.metric("Geschat aantal verhuurde fietsen", f"{int(y_pred):,}")

# --- TAB 4: Conclusions ---
with tab4:
    st.header("Conclusies & Duurzaamheid")
    st.markdown("""
    - Hogere temperaturen stimuleren fietsgebruik aanzienlijk.  
    - Metro en fiets vullen elkaar aan in drukke zones.  
    - Deze inzichten kunnen helpen om duurzaam vervoer te plannen.  
    """)

# --- TAB 5: About ---
with tab5:
    st.header("Over dit project")
    st.markdown("""
    Dit dashboard is geïnspireerd op de **NS-huisstijl** en gebaseerd op datasets over Londense mobiliteit.
    Gemaakt door **Julen Schalker** en teamgenoten voor de HvA Datascience Minor.
    """)



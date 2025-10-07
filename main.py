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
    weather = pd.read_csv("weather_london.csv")
    bike = pd.read_csv("bike_rentals.csv")

    if "tavg" in weather.columns:
        merged = pd.concat([bike["Count"], weather["tavg"]], axis=1)
        fig = px.scatter(merged, x="tavg", y="Count", trendline="ols",
                         title="Relatie tussen temperatuur en fietshuur")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("❌ Kolom 'tavg' niet gevonden in weather_london.csv")

# --- TAB 2: London Maps ---
with tab2:
    st.header("London Fiets- en Metrokaarten")
    stations = pd.read_csv("cycle_stations.csv")
    st.map(stations[["lat", "lon"]].dropna(), zoom=10)

    st.markdown("💡 Klik rechtsboven om lagen aan/uit te zetten (zoals metrostations).")

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


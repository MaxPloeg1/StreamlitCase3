# 🚆 NS: LondOnderweg! — Echte Data Dashboard (met datumfix)
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ----------------------------------------------------------
# PAGINA-INSTELLINGEN
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
# DATA INLADEN
# ----------------------------------------------------------
@st.cache_data
def load_data():
    try:
        stations = pd.read_csv("cycle_stations.csv")
        rentals = pd.read_csv("bike_rentals.csv")
        weather = pd.read_csv("weather_london.csv")
        return stations, rentals, weather
    except Exception as e:
        st.error(f"Fout bij laden van data: {e}")
        return None, None, None

stations, rentals, weather = load_data()
if stations is None or rentals is None or weather is None:
    st.error("❌ Eén of meer CSV-bestanden ontbreken. Controleer of alle data aanwezig is.")
    st.stop()

# ----------------------------------------------------------
# DATA VOORBEREIDING
# ----------------------------------------------------------
if "long" in stations.columns:
    stations = stations.rename(columns={"long": "lon"})

required_cols = ["lat", "lon", "nbBikes", "name"]
missing_cols = [col for col in required_cols if col not in stations.columns]
if missing_cols:
    st.error(f"Ontbrekende kolommen in cycle_stations.csv: {missing_cols}")
    st.stop()

lat_col, lon_col = "lat", "lon"
bike_col = "nbBikes"

# ----------------------------------------------------------
# TABSTRUCTUUR
# ----------------------------------------------------------
tab1, tab2, tab3, tab4, = st.tabs([
    "📊 Data Exploration", 
    "🚲 Fietsstations & Kaart", 
    "📈 Tijdreeks & Trends", 
    "🔮 Voorspellingsmodel"
])

# ----------------------------------------------------------
# TAB 1 — DATA EXPLORATION
# ----------------------------------------------------------
with tab1:
    st.header("Data-overzicht")
    st.markdown("Hieronder zie je voorbeelden van de drie datasets:")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🚲 cycle_stations.csv")
        st.dataframe(stations.head(), use_container_width=True)
    with c2:
        st.subheader("📅 bike_rentals.csv")
        st.dataframe(rentals.head(), use_container_width=True)
    with c3:
        st.subheader("🌦️ weather_london.csv")
        st.dataframe(weather.head(), use_container_width=True)


    st.subheader("🌦️ Weer tegenover fietsverhuringen")

    # Data voorbereiden
    rentals["Start Date"] = pd.to_datetime(rentals["Start Date"], errors="coerce")
    rentals["date"] = rentals["Start Date"].dt.date
    rentals["date"] = pd.to_datetime(rentals["date"])
    weather.rename(columns={weather.columns[0]: "date"}, inplace=True)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

    # Fietsritten per dag
    rentals_per_day = rentals.groupby("date").size().reset_index(name="rentals")

    # Samenvoegen met weerdata
    merged = pd.merge(rentals_per_day, weather, on="date", how="inner")

    st.header("📊 Correlatie tussen Windsnelheid en Temperatuur")
        # Laad en verwerk de data
    weather = pd.read_csv("weather_london.csv")
    weather.rename(columns={weather.columns[0]: "date"}, inplace=True)
    weather["date"] = pd.to_datetime(weather["date"])

    # Verwijder rijen met missende waarden
    df_corr = weather.dropna(subset=["tavg", "wspd"])

    # Bereken regressielijn
    slope, intercept, r_value, p_value, std_err = linregress(df_corr["tavg"], df_corr["wspd"])
    line = slope * df_corr["tavg"] + intercept

    # Plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(df_corr["tavg"], df_corr["wspd"], color="teal", alpha=0.6, label="Waarnemingen")
    ax.plot(df_corr["tavg"], line, color="orange", label=f"Regressielijn (r={r_value:.2f})")
    ax.set_xlabel("Gemiddelde temperatuur (°C)")
    ax.set_ylabel("Windsnelheid (m/s)")
    ax.set_title("Correlatie tussen Windsnelheid en Temperatuur")
    ax.legend()

    st.pyplot(fig, width='content')
    st.write(f"Correlatiecoëfficiënt (r): **{r_value:.2f}**")

    st.header("📊 Correlatie tussen Neerslag en Maximale Temperatuur")

    weather = pd.read_csv("weather_london.csv")
    weather.rename(columns={weather.columns[0]: "date"}, inplace=True)
    weather["date"] = pd.to_datetime(weather["date"])

    # Verwijder rijen met missende waarden
    df_corr2 = weather.dropna(subset=["prcp", "tmax"])

    # Bereken regressielijn
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(df_corr2["tmax"], df_corr2["prcp"])
    line2 = slope2 * df_corr2["tmax"] + intercept2

    # Plot
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.scatter(df_corr2["tmax"], df_corr2["prcp"], color="navy", alpha=0.6, label="Waarnemingen")
    ax2.plot(df_corr2["tmax"], line2, color="red", label=f"Regressielijn (r={r_value2:.2f})")
    ax2.set_xlabel("Maximale temperatuur (°C)")
    ax2.set_ylabel("Neerslag (mm)")
    ax2.set_title("Correlatie tussen Neerslag en Maximale Temperatuur")
    ax2.legend()

    st.pyplot(fig2, fig, width='content')
    st.write(f"Correlatiecoëfficiënt (r): **{r_value2:.2f}**")
# ----------------------------------------------------------
# TAB 2 — INTERACTIEVE KAART MET KLEURCODES
# ----------------------------------------------------------
with tab2:
    st.header("🚲 Interactieve Fietsstations Kaart")

    color_option = st.selectbox(
        "Kleurcode gebaseerd op:",
        ["nbBikes", "nbEmptyDocks", "nbDocks", "Locatie (lat/lon)"]
    )

    min_bikes = st.slider("Minimum aantal fietsen:", 0, int(stations['nbBikes'].max()), 0)
    max_bikes = st.slider("Maximum aantal fietsen:", 0, int(stations['nbBikes'].max()), int(stations['nbBikes'].max()))

    filtered_stations = stations[
        (stations['nbBikes'] >= min_bikes) & 
        (stations['nbBikes'] <= max_bikes)
    ]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gemiddeld aantal fietsen", f"{filtered_stations[bike_col].mean():.1f}")
    with col2:
        st.metric("Totaal aantal fietsen", f"{filtered_stations[bike_col].sum():,}")
    with col3:
        st.metric("Stations met 0 fietsen", f"{(filtered_stations[bike_col] == 0).sum()}")
    with col4:
        st.metric("Max fietsen per station", f"{filtered_stations[bike_col].max()}")

    # Plotly kaart
    fig_map = px.scatter_mapbox(
        filtered_stations,
        lat="lat",
        lon="lon",
        color=color_option if color_option != "Locatie (lat/lon)" else "lat",
        size="nbBikes",
        hover_name="name",
        hover_data=["nbBikes", "nbEmptyDocks", "nbDocks"],
        color_continuous_scale="Viridis",
        zoom=11,
        height=600,
        title=f"Fietsstations gekleurd op {color_option}"
    )
    fig_map.update_layout(mapbox_style="open-street-map", font=dict(color="white"))
    st.plotly_chart(fig_map, use_container_width=True)

    st.header("🔥 Drukste Routes & Gebieden")
    
    try:
        # Test if we can access the data
        st.write("Analyzing routes...")
        
        # Bereken de drukste routes
        route_counts = rentals.groupby(
            ["StartStation Id", "EndStation Id", "StartStation Name", "EndStation Name"]
        ).size().reset_index(name="trips")
        
        # Get top 10 routes
        top_routes = route_counts.nlargest(10, "trips")
        
        # Show basic table first
        st.subheader("Top 10 Drukste Routes")
        st.dataframe(top_routes[["StartStation Name", "EndStation Name", "trips"]])
        
        # Create map
        st.subheader("Kaart van Drukste Routes")
        
        # Get station coordinates
        if "id" not in stations.columns:
            stations["id"] = stations.index
        stations_dict = stations.set_index("id")[["lat", "lon", "name"]].to_dict("index")
        
        # Create base map
        center_lat = stations["lat"].mean()
        center_lon = stations["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add busiest routes
        for _, route in top_routes.iterrows():
            try:
                start_id = int(route["StartStation Id"])
                end_id = int(route["EndStation Id"])
                
                if start_id in stations_dict and end_id in stations_dict:
                    start = stations_dict[start_id]
                    end = stations_dict[end_id]
                    
                    # Draw line between stations
                    folium.PolyLine(
                        locations=[
                            [start["lat"], start["lon"]],
                            [end["lat"], end["lon"]]
                        ],
                        color="blue",
                        weight=3,
                        popup=f"{route['StartStation Name']} → {route['EndStation Name']}: {route['trips']} trips"
                    ).add_to(m)
                    
                    # Mark stations
                    for station in [start, end]:
                        folium.CircleMarker(
                            location=[station["lat"], station["lon"]],
                            radius=8,
                            color="red",
                            fill=True,
                            popup=station["name"]
                        ).add_to(m)
            
            except Exception as e:
                st.error(f"Error plotting route: {e}")
                continue
        
        # Display the map
        st_folium(m, width=800, height=600)
        
    except Exception as e:
        st.error(f"Error in tab 5: {e}")
        st.write("Debug info:")
        st.write("Rentals columns:", rentals.columns.tolist())
        st.write("Stations columns:", stations.columns.tolist())

 #TAB 3 — TIJDREEKS & TRENDS + METROKAART
# ----------------------------------------------------------
with tab3:
    st.header("📈 Tijdreeks Analyse & Metrokaart")

    # Keuzemenu voor weergave
    view_option = st.radio("📊 Kies visualisatie", ["Fietsverhuur over tijd", "Correlatie met temperatuur", "Metrokaart"], horizontal=True)

    # Voorbereiden data
    rentals["Start Date"] = pd.to_datetime(rentals["Start Date"], format="%d/%m/%Y %H:%M", errors="coerce")
    rentals["date"] = rentals["Start Date"].dt.normalize()
    rentals_per_day = rentals.groupby("date").size().reset_index(name="rentals")

    if "Unnamed: 0" in weather.columns:
        weather["date"] = pd.to_datetime(weather["Unnamed: 0"], errors="coerce")
    elif "date" in weather.columns:
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

    weather_data = weather.merge(rentals_per_day, on="date", how="inner").dropna(subset=["rentals"])

    if view_option == "Fietsverhuur over tijd":
        st.subheader("📆 Dagelijkse fietsverhuringen")
        fig_line = px.line(
            weather_data,
            x="date",
            y="rentals",
            title="Dagelijkse fietsverhuringen (observaties)",
            labels={"date": "Datum", "rentals": "Aantal verhuringen"},
            markers=True
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig_line, use_container_width=True)

    elif view_option == "Correlatie met temperatuur":
        st.subheader("🌡️ Correlatie tussen temperatuur en verhuringen")
        fig_scatter = px.scatter(
            weather_data,
            x="tavg",
            y="rentals",
            trendline="ols",
            color="rentals",
            color_continuous_scale="Viridis",
            labels={"tavg": "Gem. temperatuur (°C)", "rentals": "Verhuringen"},
            title="Relatie tussen temperatuur en aantal verhuringen"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    elif view_option == "Metrokaart":
        st.subheader("🚇 Metrokaart van Londen")

        try:
            # Laad metrodata
            stations_df = pd.read_csv("London stations.csv")
            lines_df = pd.read_csv("London tube lines.csv")

            coord_dict = stations_df.set_index("Station")[["Latitude", "Longitude"]].to_dict("index")

            tube_colors = {
                "Bakerloo": "saddlebrown",
                "Central": "red",
                "Circle": "gold",
                "District": "green",
                "Hammersmith & City": "pink",
                "Jubilee": "grey",
                "Metropolitan": "purple",
                "Northern": "black",
                "Piccadilly": "blue",
                "Victoria": "deepskyblue",
                "Waterloo & City": "turquoise",
                "DLR": "lime",
                "Overground": "orange",
                "Elizabeth": "mediumslateblue"
            }

            show_stations = st.checkbox("Toon metrostations", value=True)
            show_lines = st.checkbox("Toon metrolijnen", value=True)

            map_center = [51.5074, -0.1278]
            metro_map = folium.Map(location=map_center, zoom_start=11, tiles="cartodbpositron")

            if show_lines:
                for _, row in lines_df.iterrows():
                    from_station = row["From Station"]
                    to_station = row["To Station"]
                    line = row["Tube Line"]
                    if from_station in coord_dict and to_station in coord_dict:
                        coords = [
                            (coord_dict[from_station]["Latitude"], coord_dict[from_station]["Longitude"]),
                            (coord_dict[to_station]["Latitude"], coord_dict[to_station]["Longitude"])
                        ]
                        folium.PolyLine(
                            coords,
                            color=tube_colors.get(line, "blue"),
                            weight=3,
                            tooltip=line
                        ).add_to(metro_map)

            if show_stations:
                for station, loc in coord_dict.items():
                    folium.CircleMarker(
                        location=[loc["Latitude"], loc["Longitude"]],
                        radius=4,
                        color="green",
                        fill=True,
                        popup=station
                    ).add_to(metro_map)

            st_folium(metro_map, width=1000, height=600)

        except Exception as e:
            st.error(f"Fout bij laden metrokaart: {e}")
#voeg de grafiek toe in deze code en haal deze 2 
#Fietsverhuur over tijd

#Correlatie met temperatuur weg
# ----------------------------------------------------------
# TAB 4 — VOORSPELLINGEN MET MACHINE LEARNING (ALLEEN ÉCHTE DATA + DATUMFIX)
# ----------------------------------------------------------

# ----------------------------------------------------------
# TAB 4 — VOORSPELLINGSMODEL
# ----------------------------------------------------------
with tab4:
    st.header("🔮 Voorspellingsmodel")

    st.markdown("""
    Met dit model kun je voorspellen hoeveel fietsen er op een dag verhuurd zullen worden, 
    op basis van temperatuur, neerslag en windsnelheid. 
    Pas de waarden hieronder aan en bekijk de voorspelling.
    """)

    # -----------------------------
    # Data voorbereiden
    # -----------------------------
    rentals["Start Date"] = pd.to_datetime(rentals["Start Date"], errors="coerce")
    rentals["date"] = rentals["Start Date"].dt.date
    rentals["date"] = pd.to_datetime(rentals["date"], errors="coerce")

    # Weerdata opschonen
    weather.rename(columns={weather.columns[0]: "date"}, inplace=True)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

    for col in ["tavg", "prcp", "wspd"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather.fillna(weather.mean(numeric_only=True), inplace=True)

    # Combineer datasets
    rentals_per_day = rentals.groupby("date").size().reset_index(name="rentals")
    merged = pd.merge(rentals_per_day, weather, on="date", how="inner")

    # Check vereiste kolommen
    required_cols = ["tavg", "prcp", "wspd", "rentals"]
    if not all(col in merged.columns for col in required_cols):
        st.error("❌ Vereiste kolommen ontbreken in de data.")
        st.stop()

    # Verwijder ongeldige data
    merged = merged[required_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if merged.empty:
        st.error("❌ Onvoldoende data om model te trainen.")
        st.stop()

    # -----------------------------
    # Model trainen
    # -----------------------------
    X = merged[["tavg", "prcp", "wspd"]]
    y = merged["rentals"]

    # Gebruik een niet-lineair model (beter voor realistische verbanden)
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=8
    )
    model.fit(X, y)

    # -----------------------------
    # Sliders voor invoer
    # -----------------------------
    st.subheader("📋 Stel de weersomstandigheden in:")

    col1, col2, col3 = st.columns(3)
    with col1:
        tavg = int(st.slider("Gemiddelde temperatuur (°C)", -5, 35, int(round(merged["tavg"].mean()))))
    with col2:
        prcp = int(st.slider("Neerslag (mm)", 0, 20, int(round(merged["prcp"].mean()))))
    with col3:
        wspd = int(st.slider("Windsnelheid (m/s)", 0, 15, int(round(merged["wspd"].mean()))))

    # -----------------------------
    # Voorspelling maken
    # -----------------------------
    input_data = np.array([[tavg, prcp, wspd]])
    prediction = model.predict(input_data)[0]

    # Geen negatieve waarden toestaan + afronden
    prediction_int = max(0, int(round(prediction)))

    # -----------------------------
    # Resultaat tonen
    # -----------------------------
    st.markdown("---")
    st.subheader("📈 Verwachte fietsverhuringen")
    st.markdown(f"<h1 style='text-align:center; color:#FFD700;'>{prediction_int:,}</h1>", unsafe_allow_html=True)

    # Modelprestaties
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    st.markdown(f"**Modelprestatie:** R² = {r2:.2f} | MAE = {mae:.0f}")

    # Toon ook coëfficiëntenbelang
    st.markdown("### 📊 Belangrijkste invloeden op de voorspelling:")
    importances = pd.Series(model.feature_importances_, index=["Temperatuur", "Neerslag", "Wind"])
    st.bar_chart(importances)





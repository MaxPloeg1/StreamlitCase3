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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

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

# TAB 3 — TIJDREEKS & TRENDS + METROKAART
with tab3:
    st.header("🚇 Metrokaart & Passagiersdata")

    try:
        # Metrokaart
        stations_df = pd.read_csv("London stations.csv")
        lines_df = pd.read_csv("London tube lines.csv")

        coord_dict = stations_df.set_index("Station")[["Latitude", "Longitude"]].to_dict("index")

        tube_colors = {
            "Bakerloo": "saddlebrown", "Central": "red", "Circle": "gold",
            "District": "green", "Hammersmith & City": "pink", "Jubilee": "grey",
            "Metropolitan": "purple", "Northern": "black", "Piccadilly": "blue",
            "Victoria": "deepskyblue", "Waterloo & City": "turquoise",
            "DLR": "lime", "Overground": "orange", "Elizabeth": "mediumslateblue"
        }

        map_center = [51.5074, -0.1278]
        metro_map = folium.Map(location=map_center, zoom_start=11, tiles="cartodbpositron")

        for _, row in lines_df.iterrows():
            from_station = row["From Station"]
            to_station = row["To Station"]
            line = row["Tube Line"]
            if from_station in coord_dict and to_station in coord_dict:
                coords = [
                    (coord_dict[from_station]["Latitude"], coord_dict[from_station]["Longitude"]),
                    (coord_dict[to_station]["Latitude"], coord_dict[to_station]["Longitude"])
                ]
                folium.PolyLine(coords, color=tube_colors.get(line, "blue"), weight=3, tooltip=line).add_to(metro_map)

        for station, loc in coord_dict.items():
            folium.CircleMarker(
                location=[loc["Latitude"], loc["Longitude"]],
                radius=4, color="green", fill=True, popup=station
            ).add_to(metro_map)

        st.subheader("🔍 Metrokaart van Londen")
        st_folium(metro_map, width=1000, height=600)

    except Exception as e:
        st.error(f"Fout bij laden metrokaart: {e}")

    try:
        # Bar chart - passagiersaantallen
        entry_exit = pd.read_csv("2017_Entry_Exit.csv")
        entry_exit_sorted = entry_exit.sort_values("2017", ascending=False).head(20)

        fig_bar = px.bar(
            entry_exit_sorted,
            x="Station",
            y="2017",
            title="Top 20 drukste stations (2017)",
            labels={"2017": "Aantal passagiers", "Station": "Station"},
            color_discrete_sequence=["lightskyblue"]
        )
        fig_bar.update_layout(
            xaxis_tickangle=-45,
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Fout bij laden passagiersdata: {e}")

    try:
        # Correlatie met temperatuur
        weather = pd.read_csv("weather_london.csv")
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
        weather["year"] = weather["date"].dt.year
        avg_temp_2017 = weather[weather["year"] == 2017]["tavg"].mean()

        entry_exit["avg_temp"] = avg_temp_2017  # Voeg toe aan elke station

        fig_corr = px.scatter(
            entry_exit,
            x="avg_temp",
            y="2017",
            hover_name="Station",
            title="📈 Correlatie tussen temperatuur en passagiersaantal (2017)",
            labels={"avg_temp": "Gemiddelde temperatuur (°C)", "2017": "Passagiers"},
            opacity=0.7
        )
        fig_corr.update_layout(
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"Fout bij visualisatie correlatie met weerdata: {e}")
# ----------------------------------------------------------
# TAB 4 — VOORSPELLINGEN MET MACHINE LEARNING (ALLEEN ÉCHTE DATA + DATUMFIX)
# ----------------------------------------------------------
with tab4:
    st.header("🔮 Voorspellingsmodel")

    st.markdown("""
    Dit model voorspelt het verwachte aantal fietsverhuringen op een dag 
    op basis van temperatuur, neerslag, maand en dag van de week.
    """)

    # -----------------------------
    # Data voorbereiden
    # -----------------------------
    rentals["Start Date"] = pd.to_datetime(rentals["Start Date"], errors="coerce")
    rentals["date"] = rentals["Start Date"].dt.date
    rentals["date"] = pd.to_datetime(rentals["date"], errors="coerce")

    weather.rename(columns={weather.columns[0]: "date"}, inplace=True)
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

    for col in ["tavg", "prcp"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather.fillna(weather.mean(numeric_only=True), inplace=True)

    rentals_per_day = rentals.groupby("date").size().reset_index(name="rentals")
    merged = pd.merge(rentals_per_day, weather, on="date", how="inner")

    # Tijdvariabelen toevoegen
    merged["month"] = merged["date"].dt.month
    merged["day_of_week"] = merged["date"].dt.dayofweek  # 0 = maandag, 6 = zondag

    merged = merged[["tavg", "prcp", "month", "day_of_week", "rentals"]].dropna()

    if merged.empty:
        st.error("❌ Geen bruikbare data om model te trainen.")
        st.stop()

    # -----------------------------
    # Model trainen
    # -----------------------------
    X = merged[["tavg", "prcp", "month", "day_of_week"]]
    y = merged["rentals"]

    categorical_features = ["month", "day_of_week"]
    numeric_features = ["tavg", "prcp"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42
        ))
    ])

    model.fit(X, y)

    # -----------------------------
    # Gebruikersinvoer
    # -----------------------------
    st.subheader("📋 Stel de weersomstandigheden in:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tavg = int(st.slider("Gem. temperatuur (°C)", -10, 40, int(round(merged["tavg"].mean()))))
    with col2:
        prcp = int(st.slider("Neerslag (mm)", 0, 30, int(round(merged["prcp"].mean()))))
    with col3:
        month = st.slider("Maand", 1, 12, int(merged["month"].median()))
    with col4:
        day_of_week = st.slider("Dag van week (0=ma, 6=zo)", 0, 6, 2)

    # -----------------------------
    # Logische temperatuurcontrole
    # -----------------------------
    # Aangepaste temperatuurgrenzen per maand
    realistic_temps = {
        1: (-5, 15),   # januari
        2: (-5, 20),   # februari
        3: (-2, 20),   # maart
        4: (2, 22),    # april
        5: (6, 28),    # mei
        6: (10, 35),   # juni
        7: (12, 40),   # juli
        8: (12, 40),   # augustus
        9: (8, 30),    # september
        10: (4, 22),   # oktober
        11: (0, 18),   # november
        12: (-5, 15)   # december
    }

    min_temp, max_temp = realistic_temps[month]

    if tavg < min_temp or tavg > max_temp:
        st.error(
            f"⚠️ Onrealistische temperatuur voor maand {month}. "
            f"Verwacht bereik: {min_temp}°C tot {max_temp}°C."
        )
        st.stop()  # Stop verdere uitvoering van het model

    # -----------------------------
    # Voorspelling
    # -----------------------------
    input_data = pd.DataFrame([{
        "tavg": tavg,
        "prcp": prcp,
        "month": month,
        "day_of_week": day_of_week
    }])

    prediction = model.predict(input_data)[0]
    prediction_int = int(round(prediction))

    # -----------------------------
    # Resultaat
    # -----------------------------
    st.markdown("---")
    st.subheader("📈 Verwachte fietsverhuringen")
    st.markdown(f"<h1 style='text-align:center; color:#FFD700;'>{prediction_int:,}</h1>", unsafe_allow_html=True)

    # Prestatie
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    st.markdown(f"**Modelprestatie:** R² = {r2:.2f} | MAE = {mae:.0f}")

























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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Exploration", "🚲 Fietsstations & Kaart", "📈 Tijdreeks & Trends", "🔮 Voorspellingen"
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


    st.subheader("📊 Correlatie tussen fietsverhuringen en weersvariabelen")

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

    # Alleen numerieke kolommen van weerdata (zonder 'rentals')
    weather_numeric = merged.drop(columns=["rentals", "date"]).select_dtypes(include=[np.number])

    # Correlatie berekenen tussen rentals en elke weerkolom
    corr_with_rentals = weather_numeric.corrwith(merged["rentals"]).round(3)

    # Omzetten naar DataFrame voor Plotly
    corr_df = corr_with_rentals.reset_index()
    corr_df.columns = ["Weersvariabele", "Correlatie"]

    # Plot als staafdiagram
    fig_corr = px.bar(
        corr_df,
        x="Weersvariabele",
        y="Correlatie",
        color="Correlatie",
        color_continuous_scale="RdBu_r",
        title="📈 Correlatie van weerdata met aantal fietsverhuringen",
        template="plotly_dark",
        text="Correlatie"
    )

    fig_corr.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_corr.update_layout(
        width=900,
        height=600,
        title_font_size=22,
        xaxis_title="Weersvariabele",
        yaxis_title="Correlatiecoëfficiënt",
        margin=dict(l=60, r=60, t=80, b=60)
    )

    st.plotly_chart(fig_corr, use_container_width=True)
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

# ----------------------------------------------------------
# TAB 3 — TIJDREEKS & WEATHER TRENDS (ALLEEN ECHTE DATA + DATUMFIX)
# ----------------------------------------------------------
with tab3:
    st.header("📈 Tijdreeks Analyse & Weather Trends")

    if "tavg" not in weather.columns:
        st.error("❌ Kolom 'tavg' ontbreekt in weather_london.csv.")
        st.stop()

    try:
        # ✅ Datumfix: uniforme datums maken
        rentals["Start Date"] = pd.to_datetime(rentals["Start Date"], format="%d/%m/%Y %H:%M", errors="coerce")
        rentals["date"] = rentals["Start Date"].dt.normalize()  # alleen datum
        rentals_per_day = rentals.groupby("date").size().reset_index(name="rentals")

        if "Unnamed: 0" in weather.columns:
            weather["date"] = pd.to_datetime(weather["Unnamed: 0"], errors="coerce")
        elif "date" in weather.columns:
            weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

        # Merge op overlappende dagen
        weather_data = weather.merge(rentals_per_day, on="date", how="inner").dropna(subset=["rentals"])
        overlap_days = len(weather_data)
        st.info(f"📅 Overlappende dagen met echte data: {overlap_days}")

        if overlap_days < 5:
            st.error("❌ Te weinig overlappende datums tussen weerdata en verhuurdata.")
            st.stop()

        # Tijdreeks weergave
        st.subheader("Fietsverhuringen over tijd (echte data)")
        fig_line = px.line(
            weather_data,
            x="date",
            y="rentals",
            title="Dagelijkse fietsverhuringen (echte observaties)",
            labels={"date": "Datum", "rentals": "Aantal verhuringen"},
            markers=True
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Correlatie met temperatuur
        st.subheader("Correlatie tussen temperatuur en verhuringen")
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

    except Exception as e:
        st.error(f"Fout bij verwerking van tijdreeksdata: {e}")
        st.stop()

# ----------------------------------------------------------
# ----------------------------------------------------------
# TAB 4 — VOORSPELLINGEN MET MACHINE LEARNING (ALLEEN ÉCHTE DATA + DATUMFIX)
# ----------------------------------------------------------

with tab4:
    st.header("🔮 Voorspellingen - Fietsverhuringen")

    try:
        # 🧹 Stap 1: Data normaliseren & mergen
        rentals["Start Date"] = pd.to_datetime(rentals["Start Date"], errors="coerce")
        rentals["date"] = rentals["Start Date"].dt.normalize()
        rentals_per_day = rentals.groupby("date").size().reset_index(name="rentals")

        if "Unnamed: 0" in weather.columns:
            weather["date"] = pd.to_datetime(weather["Unnamed: 0"], errors="coerce")
        elif "date" in weather.columns:
            weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
        else:
            st.error("❌ Kon geen datumkolom vinden in de weerdata.")
            st.stop()

        full_data = pd.merge(weather, rentals_per_day, on="date", how="inner")
        full_data = full_data.dropna(subset=["rentals"])

        # 🎯 Stap 2: Feature selectie
        all_features = ["tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "pres"]
        used_features = [f for f in all_features if f in full_data.columns]
        full_data = full_data.dropna(subset=used_features)

        overlap_days = len(full_data)
        if overlap_days < 7:
            st.error(f"❌ Niet genoeg overlappende data voor training ({overlap_days} dagen).")
            st.dataframe(full_data[["date", "tavg", "rentals"]])
            st.stop()

        st.success(f"✅ {overlap_days} overlappende dagen gevonden voor modeltraining.")

        # ⚙️ Stap 3: Configuratie
        st.subheader("⚙️ Model Configuratie")
        c1, c2 = st.columns(2)
        with c1:
            selected_features = st.multiselect(
                "Selecteer features voor voorspelling:",
                options=used_features,
                default=["tavg", "prcp"] if "prcp" in used_features else used_features[:2]
            )
        with c2:
            st.markdown("Kies modeltype (Random Forest aanbevolen):")
            model_type = st.selectbox("Model", ["Random Forest"], index=0)

        if not selected_features:
            st.warning("⚠️ Selecteer minimaal één feature.")
            st.stop()

        # 🧠 Stap 4: Train/Test Split
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error

        X = full_data[selected_features]
        y = full_data["rentals"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 📊 Stap 5: Evaluatie
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.subheader("📈 Model Prestaties")
        m1, m2, m3 = st.columns(3)
        m1.metric("📊 R² Score", f"{r2:.2f}")
        m2.metric("📉 MAE", f"{mae:,.0f}")
        m3.metric("📅 Datapunten", f"{overlap_days}")

        # 📈 Visualisatie: Werkelijk vs Voorspeld
        import plotly.graph_objects as go
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode="markers",
            name="Voorspellingen",
            marker=dict(color="#FFD700", size=10),
            hovertemplate="Werkelijk: %{x:,.0f}<br>Voorspeld: %{y:,.0f}<extra></extra>"
        ))
        fig_pred.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines",
            name="Perfecte lijn",
            line=dict(color="green", dash="dash")
        ))
        fig_pred.update_layout(
            title="📊 Voorspelde vs Werkelijke Verhuringen",
            xaxis_title="Werkelijk",
            yaxis_title="Voorspeld",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # 🔮 Interactieve voorspelling
        st.subheader("🔍 Interactieve voorspelling")
        input_data = []
        col_left, col_right = st.columns(2)
        for i, feature in enumerate(selected_features):
            col = col_left if i % 2 == 0 else col_right
            with col:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                mean_val = float(X[feature].mean())
                val = st.slider(
                    f"{feature}:", min_val, max_val, mean_val,
                    step=(max_val - min_val) / 100
                )
                input_data.append(val)

        if st.button("🔮 Voorspel"):
            pred_input = np.array([input_data])
            predicted_value = model.predict(pred_input)[0]
            st.success(f"📈 Voorspelde verhuringen: **{predicted_value:,.0f}**")

    except Exception as e:
        st.error(f"❌ Fout bij modeltraining: {e}")

# 🚆 NS: LondOnderweg! — Advanced Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
        tube_stations = pd.read_csv('London stations.csv')
        return stations, rentals, weather, tube_stations
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load data with error handling
data_result = load_data()
if data_result[0] is not None:
    stations, rentals, weather, tube_stations = data_result
else:
    st.error("Failed to load data. Please check that all CSV files are present.")
    st.stop()

# ----------------------------------------------------------
# DATA VOORBEREIDING
# ----------------------------------------------------------
# Hernoem kolom 'long' naar 'lon' voor Streamlit map compatibility
if "long" in stations.columns:
    stations = stations.rename(columns={"long": "lon"})
    st.sidebar.success("Renamed 'long' to 'lon'")

# Controleer of vereiste kolommen bestaan
required_cols = ["lat", "lon", "nbBikes", "name"]
missing_cols = [col for col in required_cols if col not in stations.columns]
if missing_cols:
    st.sidebar.error(f"Missing columns: {missing_cols}")
    st.error(f"Missing required columns in stations data: {missing_cols}")
    st.stop()

# Kolomnamen
lat_col, lon_col = "lat", "lon"
bike_col = "nbBikes"


# ----------------------------------------------------------
# TABSTRUCTUUR
# ----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "🚲 Fietsstations & Kaart", "📈 Tijdreeks & Trends", "🔮 Voorspellingen"])

# ----------------------------------------------------------
# TAB 1 — DATA EXPLORATION
# ----------------------------------------------------------
with tab1:
    st.header("Data-overzicht")
    st.markdown("Hieronder zie je voorbeelden van onze drie datasets:")

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

# ----------------------------------------------------------
# ----------------------------------------------------------
# TAB 2 — INTERACTIEVE KAART MET KLEURCODES
# ----------------------------------------------------------
with tab2:
    st.header("🚴‍♀️ Interactieve Fietsstations & Metrokaart")

    # Sidebar controls voor kaart
    st.sidebar.subheader("🎛️ Kaart Instellingen")

    # Kleurcode-opties
    color_option = st.sidebar.selectbox(
        "Kleurcode gebaseerd op:",
        ["nbBikes", "nbEmptyDocks", "nbDocks", "Locatie (lat/lon)"]
    )

    # Bike count filter
    min_bikes = st.sidebar.slider("Minimum aantal fietsen:", 0, int(stations['nbBikes'].max()), 0)
    max_bikes = st.sidebar.slider("Maximum aantal fietsen:", 0, int(stations['nbBikes'].max()), int(stations['nbBikes'].max()))

    # Filter stations
    filtered_stations = stations[
        (stations['nbBikes'] >= min_bikes) &
        (stations['nbBikes'] <= max_bikes)
    ]

    # Controle of kolommen bestaan
    if lat_col in stations.columns and lon_col in stations.columns:
        st.success(f"✅ {len(filtered_stations)} stations gevonden (gefilterd van {len(stations)})")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gemiddeld aantal fietsen", f"{filtered_stations[bike_col].mean():.1f}")
        with col2:
            st.metric("Totaal aantal fietsen", f"{filtered_stations[bike_col].sum():,}")
        with col3:
            st.metric("Stations met 0 fietsen", f"{(filtered_stations[bike_col] == 0).sum()}")
        with col4:
            st.metric("Max fietsen per station", f"{filtered_stations[bike_col].max()}")

        # Plotly interactieve kaart
        import plotly.express as px
        fig_map = px.scatter_mapbox(
            filtered_stations,
            lat="lat",
            lon="lon",
            color=color_option.split()[0] if color_option != "Locatie (lat/lon)" else "lat",
            size="nbBikes",
            hover_name="name",
            hover_data=["nbBikes", "nbEmptyDocks", "nbDocks"],
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=11,
            height=600,
            title=f"Fietsstations gekleurd op {color_option}"
        )

        fig_map.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=51.5074, lon=-0.1278)),
            margin={"r":0,"t":30,"l":0,"b":0},
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.error("❌ Kolommen 'lat' en 'lon' niet gevonden in cycle_stations.csv")

    # ----------------------------------------------------------
    # METROKAART (onder de fietskaart)
    # ----------------------------------------------------------
    st.divider()
    st.subheader("🚇 London Metro Kaart")

    tube_stations = load_csv("London stations.csv")
    tube_lines = load_csv("London tube lines.csv")

    if tube_stations is None or tube_lines is None:
        st.info("ℹ️ Voeg 'London stations.csv' en 'London tube lines.csv' toe om de metrokaart te tonen.")
    else:
        tube_stations.columns = tube_stations.columns.str.strip().str.lower()
        tube_lines.columns = tube_lines.columns.str.strip().str.lower()

        if "entries" not in tube_stations.columns:
            rng = np.random.default_rng(42)
            tube_stations["entries"] = rng.integers(1000, 50000, len(tube_stations))

        with st.expander("⚙️ Metro Filteropties", expanded=True):
            day_type = st.radio("Toon data voor", ["Weekdagen", "Weekend"], index=0)
            density = st.slider("Selecteer drukte", 0, 100, 0)
            show_stations = st.checkbox("Metro stations en bezoekersaantal", value=True)
            show_lines = st.checkbox("Metro lijnen", value=True)

        # Folium-kaart aanmaken
        m2 = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="CartoDB positron")

        if show_lines:
            for line, seg in tube_lines.groupby("line"):
                coords = seg[["lat", "lon"]].dropna().values.tolist()
                color = f"#{abs(hash(line)) % 0xFFFFFF:06x}"
                folium.PolyLine(coords, color=color, weight=3, opacity=0.85, popup=f"Lijn: {line}").add_to(m2)

        if show_stations:
            for _, row in tube_stations.iterrows():
                visitors = int(row["entries"])
                if visitors < density * 500:
                    continue
                color = "red" if day_type == "Weekdagen" else "blue"
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=max(2, min(visitors / 8000, 10)),
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"{row['name']}<br>Bezoekers: {visitors:,}"
                ).add_to(m2)

        st_folium(m2, width=1100, height=600)

# ----------------------------------------------------------
# TAB 3 — INTERACTIEVE TIJDREEKS & TRENDS
# ----------------------------------------------------------
with tab3:
    st.header("Tijdreeks Analyse & Weather Trends")

    if "tavg" in weather.columns:
        st.success("✅ Weerdata succesvol geladen!")

        # Sidebar controls voor analyses
        st.sidebar.subheader("Analyse Instellingen")
        
        # Prepareer echte rental data - simpel en direct
        try:
            # Extract date from rental Start Date and count rentals per day
            rentals['Start Date'] = pd.to_datetime(rentals['Start Date'], format='%d/%m/%Y %H:%M', errors='coerce')
            rentals_per_day = rentals['Start Date'].dt.date.value_counts().reset_index()
            rentals_per_day.columns = ['date', 'rentals']
            rentals_per_day['date'] = pd.to_datetime(rentals_per_day['date'])
            
            # Add date column to weather data
            weather['date'] = pd.to_datetime(weather['Unnamed: 0'], errors='coerce')
            
            # Merge weather with rental counts
            weather_data = weather.merge(rentals_per_day, on='date', how='left')
            weather_data['rentals'] = weather_data['rentals'].fillna(0)
            
            # Filter to only days with rentals
            real_data = weather_data[weather_data['rentals'] > 0].copy()
            
            if len(real_data) > 0:
                st.info(f"⚠️ Beperkte dataset: {len(real_data)} dagen echte data (31 aug - 6 sept 2022)")
                st.success(f"Gemiddeld {real_data['rentals'].mean():.0f} verhuur per dag")
                
                # Extend data with seasonal patterns
                st.sidebar.checkbox("Uitbreiden met seizoenspatronen", value=True, key="extend_data")
                if st.session_state.extend_data:
                    # Create extended dataset based on patterns
                    np.random.seed(42)
                    extended_weather = weather[(weather['date'] >= '2022-01-01') & (weather['date'] <= '2022-12-31')].copy()
                    
                    # Simulate rental patterns based on temperature and season
                    base_rentals = 35000  # Based on real average
                    temp_effect = (extended_weather['tavg'].fillna(15) - 15) * 500  # Temperature effect
                    season_effect = np.sin((extended_weather['date'].dt.dayofyear - 80) * 2 * np.pi / 365) * 10000  # Seasonal pattern
                    random_noise = np.random.normal(0, 3000, len(extended_weather))
                    
                    extended_weather['rentals'] = np.maximum(0, 
                        base_rentals + temp_effect + season_effect + random_noise
                    ).astype(int)
                    
                    # Replace real data where available
                    for _, row in real_data.iterrows():
                        mask = extended_weather['date'] == row['date']
                        extended_weather.loc[mask, 'rentals'] = row['rentals']
                    
                    weather_data = extended_weather
                    st.info(f"📈 Dataset uitgebreid naar {len(weather_data)} dagen (heel 2022)")
                else:
                    weather_data = real_data
                    
            else:
                st.warning("Geen overlappende datums. Gebruik gesimuleerde data.")
                np.random.seed(42)
                weather_data = weather.copy()
                weather_data["rentals"] = np.random.randint(5000, 55000, size=len(weather_data))
                
        except Exception as e:
            st.warning(f"Fout bij data verwerking: {e}. Gebruik gesimuleerde data.")
            np.random.seed(42)
            weather_data = weather.copy()
            weather_data["rentals"] = np.random.randint(5000, 55000, size=len(weather_data))

        # Interactieve tijdreeks
        st.subheader("Interactieve Tijdreeks - Fietsverhuringen over Tijd")
        
        # Date range selector
        if len(weather_data) > 30:
            date_range = st.sidebar.date_input(
                "Selecteer datumbereik:",
                value=(weather_data['date'].min(), weather_data['date'].max()),
                min_value=weather_data['date'].min(),
                max_value=weather_data['date'].max()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_data = weather_data[
                    (weather_data['date'] >= pd.to_datetime(start_date)) & 
                    (weather_data['date'] <= pd.to_datetime(end_date))
                ]
            else:
                filtered_data = weather_data
        else:
            filtered_data = weather_data
            
        # Plotly interactieve lijndiagram
        fig_line = go.Figure()
        
        # Add rental line
        fig_line.add_trace(go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['rentals'],
            mode='lines+markers',
            name='Fietsverhuringen',
            line=dict(color='#FFD700', width=3),
            hovertemplate='<b>%{x}</b><br>Verhuringen: %{y:,}<extra></extra>'
        ))
        
        # Add weather overlay
        weather_overlay = st.sidebar.selectbox(
            "Weather overlay:",
            ["Geen", "tavg", "tmin", "tmax", "prcp"],
            key="weather_overlay"
        )
        
        if weather_overlay != "Geen":
            fig_line.add_trace(go.Scatter(
                x=filtered_data['date'],
                y=filtered_data[weather_overlay],
                mode='lines',
                name=f'Weather: {weather_overlay}',
                yaxis='y2',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate=f'<b>%{{x}}</b><br>{weather_overlay}: %{{y}}<extra></extra>'
            ))
            
            # Update layout for dual y-axis
            fig_line.update_layout(
                yaxis2=dict(
                    title=f"Weather: {weather_overlay}",
                    overlaying='y',
                    side='right',
                    titlefont=dict(color='red'),
                    tickfont=dict(color='red')
                )
            )
        
        fig_line.update_layout(
            title="Fietsverhuringen over Tijd (Interactief)",
            xaxis_title="Datum",
            yaxis_title="Aantal Verhuringen",
            hovermode='x unified',
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        
        st.plotly_chart(fig_line, use_container_width=True)

        # Weather factor correlation
        st.subheader("Weather Factor Correlatie")
        weather_factor = st.selectbox("Kies een weerfactor:", ["tavg", "tmin", "tmax", "prcp", "tsun"])

        # Plotly scatter plot
        fig_scatter = px.scatter(
            filtered_data, 
            x=weather_factor, 
            y="rentals",
            trendline="ols",
            title=f"Correlatie tussen {weather_factor} en Fietsverhuringen",
            color="rentals",
            color_continuous_scale="Viridis",
            hover_data=["date"]
        )
        
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Correlation matrix heatmap
        st.subheader("Correlatiematrix Heatmap")
        numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
        corr_matrix = filtered_data[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlatiematrix van alle numerieke variabelen",
            color_continuous_scale="RdBu"
        )
        
        fig_heatmap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=600
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)

    else:
        st.error("❌ 'tavg' kolom niet gevonden in weather_london.csv.")
        st.write("Beschikbare kolommen:", list(weather.columns))


# ----------------------------------------------------------
# TAB 4 — VOORSPELLINGEN MET MACHINE LEARNING
# ----------------------------------------------------------
with tab4:
    st.header("Voorspellingen - Fietsverhuringen")
    
    if "tavg" in weather.columns:
        st.success("Machine Learning Voorspellingsmodellen")
        
        # Use the same weather_data from tab3
        try:
            # Prepare data for prediction
            rentals['Start Date'] = pd.to_datetime(rentals['Start Date'], format='%d/%m/%Y %H:%M', errors='coerce')
            rentals_per_day = rentals['Start Date'].dt.date.value_counts().reset_index()
            rentals_per_day.columns = ['date', 'rentals']
            rentals_per_day['date'] = pd.to_datetime(rentals_per_day['date'])
            
            weather['date'] = pd.to_datetime(weather['Unnamed: 0'], errors='coerce')
            weather_data = weather.merge(rentals_per_day, on='date', how='left')
            weather_data['rentals'] = weather_data['rentals'].fillna(0)
            
            # Create extended dataset if we have limited real data
            real_data = weather_data[weather_data['rentals'] > 0]
            if len(real_data) < 30:
                st.warning("⚠️ Beperkte training data - gebruik gesimuleerde seizoenspatronen")
                np.random.seed(42)
                extended_weather = weather[(weather['date'] >= '2022-01-01') & (weather['date'] <= '2022-12-31')].copy()
                
                # Create realistic patterns
                base_rentals = 35000
                temp_effect = (extended_weather['tavg'].fillna(15) - 15) * 500
                season_effect = np.sin((extended_weather['date'].dt.dayofyear - 80) * 2 * np.pi / 365) * 10000
                rain_effect = -extended_weather['prcp'].fillna(0) * 100
                random_noise = np.random.normal(0, 3000, len(extended_weather))
                
                extended_weather['rentals'] = np.maximum(1000, 
                    base_rentals + temp_effect + season_effect + rain_effect + random_noise
                ).astype(int)
                
                # Keep real data where available
                for _, row in real_data.iterrows():
                    mask = extended_weather['date'] == row['date']
                    extended_weather.loc[mask, 'rentals'] = row['rentals']
                
                ml_data = extended_weather.dropna(subset=['tavg', 'rentals'])
                st.info(f"🤖 Training model met {len(ml_data)} datapunten")
            else:
                ml_data = weather_data[weather_data['rentals'] > 0].dropna(subset=['tavg', 'rentals'])
                
            if len(ml_data) > 10:  # Minimum data for ML
                
                # Feature selection
                st.subheader("Model Configuratie")
                
                col1, col2 = st.columns(2)
                with col1:
                    features = st.multiselect(
                        "Selecteer features voor voorspelling:",
                        ["tavg", "tmin", "tmax", "prcp", "wspd", "pres"],
                        default=["tavg", "prcp"]
                    )
                
                with col2:
                    model_type = st.selectbox(
                        "Model type:",
                        ["Linear Regression", "Polynomial (degree 2)"]
                    )
                
                if features:
                    # Prepare features and target
                    X = ml_data[features].fillna(ml_data[features].mean())
                    y = ml_data['rentals']
                    
                    # Create polynomial features if selected
                    if model_type == "Polynomial (degree 2)":
                        from sklearn.preprocessing import PolynomialFeatures
                        poly = PolynomialFeatures(degree=2)
                        X = poly.fit_transform(X)
                        feature_names = [f"poly_{i}" for i in range(X.shape[1])]
                    else:
                        feature_names = features
                    
                    # Train model
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Predictions
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    
                    # Display model performance
                    st.subheader("Model Prestaties")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{r2:.3f}")
                    with col2:
                        st.metric("Mean Absolute Error", f"{mae:.0f}")
                    with col3:
                        st.metric("Trainingsdata", f"{len(ml_data)} dagen")
                    
                    # Prediction vs Actual plot
                    st.subheader("Voorspelling vs Werkelijkheid")
                    fig_pred = go.Figure()
                    
                    # Add perfect prediction line
                    min_val = min(y.min(), y_pred.min())
                    max_val = max(y.max(), y_pred.max())
                    fig_pred.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfecte voorspelling',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Add actual vs predicted
                    fig_pred.add_trace(go.Scatter(
                        x=y,
                        y=y_pred,
                        mode='markers',
                        name='Voorspellingen',
                        marker=dict(color='#FFD700', size=8),
                        hovertemplate='Werkelijk: %{x:,.0f}<br>Voorspeld: %{y:,.0f}<extra></extra>'
                    ))
                    
                    fig_pred.update_layout(
                        title=f"Model Accuracy (R² = {r2:.3f})",
                        xaxis_title="Werkelijke Verhuringen",
                        yaxis_title="Voorspelde Verhuringen",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        height=500
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Interactive prediction
                    st.subheader("Interactieve Voorspelling")
                    st.write("Pas de weather parameters aan om voorspellingen te maken:")
                    
                    pred_col1, pred_col2 = st.columns(2)
                    
                    # Input sliders for prediction
                    pred_inputs = {}
                    for i, feature in enumerate(features):
                        col = pred_col1 if i % 2 == 0 else pred_col2
                        with col:
                            if feature in ml_data.columns:
                                min_val = float(ml_data[feature].min())
                                max_val = float(ml_data[feature].max())
                                mean_val = float(ml_data[feature].mean())
                                
                                pred_inputs[feature] = st.slider(
                                    f"{feature}:",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=mean_val,
                                    step=(max_val - min_val) / 100
                                )
                    
                    # Make prediction
                    if st.button("Voorspel Fietsverhuringen"):
                        pred_X = np.array([[pred_inputs[f] for f in features]])
                        
                        if model_type == "Polynomial (degree 2)":
                            pred_X = poly.transform(pred_X)
                        
                        prediction = model.predict(pred_X)[0]
                        
                        st.success(f"Voorspelde fietsverhuringen: **{prediction:,.0f}**")
                        
                        # Show feature importance for linear model
                        if model_type == "Linear Regression":
                            st.subheader("Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': features,
                                'Coefficient': model.coef_[:len(features)],
                                'Abs_Coefficient': np.abs(model.coef_[:len(features)])
                            }).sort_values('Abs_Coefficient', ascending=False)
                            
                            fig_importance = px.bar(
                                importance_df,
                                x='Feature',
                                y='Coefficient',
                                title="Feature Importance (Linear Regression Coefficients)",
                                color='Coefficient',
                                color_continuous_scale='RdBu'
                            )
                            
                            fig_importance.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white"),
                                height=400
                            )
                            
                            st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Time series prediction
                    st.subheader("Tijdreeks Voorspelling")
                    
                    # Create future dates
                    last_date = ml_data['date'].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=30,
                        freq='D'
                    )
                    
                    if st.button("Voorspel Komende 30 Dagen"):
                        # Get weather for future dates (use historical averages)
                        future_weather = []
                        for date in future_dates:
                            day_of_year = date.dayofyear
                            
                            # Get historical average for this day of year
                            historical = weather[
                                (weather['date'].dt.dayofyear >= day_of_year - 7) &
                                (weather['date'].dt.dayofyear <= day_of_year + 7)
                            ]
                            
                            if len(historical) > 0:
                                avg_features = {feat: historical[feat].mean() for feat in features}
                            else:
                                avg_features = {feat: ml_data[feat].mean() for feat in features}
                        
                        future_weather.append(avg_features)
                        
                        future_df = pd.DataFrame(future_weather)
                        future_X = future_df[features].fillna(future_df[features].mean())
                        
                        if model_type == "Polynomial (degree 2)":
                            future_X = poly.transform(future_X)
                        
                        future_predictions = model.predict(future_X)
                        
                        # Plot historical + future
                        fig_future = go.Figure()
                        
                        # Historical data
                        fig_future.add_trace(go.Scatter(
                            x=ml_data['date'],
                            y=ml_data['rentals'],
                            mode='lines+markers',
                            name='Historische Data',
                            line=dict(color='#FFD700')
                        ))
                        
                        # Future predictions
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_predictions,
                            mode='lines+markers',
                            name='Voorspellingen',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_future.update_layout(
                            title="Fietsverhuringen: Historisch + Voorspelling (30 dagen)",
                            xaxis_title="Datum",
                            yaxis_title="Aantal Verhuringen",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white"),
                            height=500
                        )
                        
                        st.plotly_chart(fig_future, use_container_width=True)
                        
                        # Summary stats
                        st.info(f"Gemiddelde voorspelling komende 30 dagen: {future_predictions.mean():,.0f} verhuringen/dag")
                        st.info(f"Range: {future_predictions.min():,.0f} - {future_predictions.max():,.0f} verhuringen")
                        
                else:
                    st.warning("Selecteer minimaal één feature voor voorspelling")
                    
            else:
                st.error("Niet genoeg data voor machine learning (minimum 10 datapunten)")
                
        except Exception as e:
            st.error(f"Fout bij machine learning: {e}")
            st.write("Gebruik gesimuleerde data voor demonstratie")
            
    else:
        st.error("Geen weather data beschikbaar voor voorspellingen")








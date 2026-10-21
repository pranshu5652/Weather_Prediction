
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# 1. APP SETTINGS
# -----------------------------
st.set_page_config(
    page_title=" Weather Prediction",
    page_icon="🌤️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "weather_model.pkl"

LATITUDE = 26.82
LONGITUDE = 80.93
TIMEZONE = "Asia/Kolkata"

API_URL = "https://api.open-meteo.com/v1/forecast"


# -----------------------------
# 2. LOAD TRAINED ML MODEL
# -----------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


# -----------------------------
# 3. FETCH LIVE WEATHER + FORECAST
# -----------------------------
@st.cache_data(ttl=600)
def fetch_weather():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "apparent_temperature,precipitation,wind_speed_10m"
        ),
        "daily": (
            "temperature_2m_mean,temperature_2m_max,"
            "temperature_2m_min,precipitation_sum,"
            "wind_speed_10m_max"
        ),
        "past_days": 14,
        "forecast_days": 16
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=20
    )
    response.raise_for_status()
    return response.json()


# -----------------------------
# 4. BUILD ML PREDICTION
# -----------------------------
def predict_for_date(model, feature_names, daily, selected_date):
    # Use complete days before today as historical input.
    history_df = daily[daily["date"] < date.today()].copy()

    history_df = history_df.dropna(
        subset=["temperature_2m_mean"]
    )

    history = {
        row["date"]: float(row["temperature_2m_mean"])
        for _, row in history_df.iterrows()
    }

    if not history:
        raise ValueError("Recent temperature history is unavailable.")

    target_date = min(history)  # Placeholder replaced below
    target_date = date.today()

    if len(history) < 8:
        raise ValueError("ML model needs at least 8 daily temperature records.")

    # Generate predictions one day at a time.
    while target_date <= selected_date:
        previous_dates = sorted(
            d for d in history if d < target_date
        )

        if not previous_dates:
            raise ValueError("Previous-day data is missing.")

        latest_date = previous_dates[-1]

        # Require consecutive calendar-day observations/predictions.
        required_dates = [
            date.fromordinal(target_date.toordinal() - offset)
            for offset in range(1, 9)
        ]

        if any(d not in history for d in required_dates):
            raise ValueError(
                "Consecutive daily temperature data is missing."
            )

        input_values = {
            "target_month": target_date.month,
            "target_day_of_year": target_date.timetuple().tm_yday,
            "temp_lag_0": history[required_dates[0]],
            "temp_lag_1": history[required_dates[1]],
            "temp_lag_3": history[required_dates[3]],
            "temp_lag_7": history[required_dates[7]],
            "rolling_mean_7": sum(
                history[d] for d in required_dates[:7]
            ) / 7
        }

        input_df = pd.DataFrame(
            [input_values]
        )[feature_names]

        prediction = float(model.predict(input_df)[0])
        history[target_date] = prediction

        if target_date == selected_date:
            return prediction

        target_date = date.fromordinal(
            target_date.toordinal() + 1
        )

    return history[selected_date]


# -----------------------------
# 5. DISPLAY APP
# -----------------------------
st.title("🌤️ Live Weather Prediction")
st.caption(
    "Live conditions and date-wise temperature estimates | "
    "Data: Open-Meteo"
)

try:
    model, feature_names = load_model()
    weather = fetch_weather()

except Exception as error:
    st.error(f"Data/model load nahi hua: {error}")
    st.info(
        "Check karo ki models/weather_model.pkl exist karta hai "
        "aur internet connection available hai."
    )
    st.stop()


# Current conditions
current = weather["current"]

st.subheader("📍 Current Weather — Lucknow")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Temperature",
    f'{current["temperature_2m"]} °C'
)
c2.metric(
    "Feels Like",
    f'{current["apparent_temperature"]} °C'
)
c3.metric(
    "Humidity",
    f'{current["relative_humidity_2m"]} %'
)
c4.metric(
    "Wind Speed",
    f'{current["wind_speed_10m"]} km/h'
)

st.caption(
    f'API observation time: {current["time"]} '
    f'({TIMEZONE})'
)


# Convert daily API data to a DataFrame
daily = pd.DataFrame(weather["daily"])
daily["date"] = pd.to_datetime(
    daily["time"]
).dt.date

daily["temperature_2m_mean"] = pd.to_numeric(
    daily["temperature_2m_mean"],
    errors="coerce"
)

forecast_dates = sorted(
    d for d in daily["date"]
    if d >= date.today()
)

if not forecast_dates:
    st.error("API ne forecast dates return nahi ki.")
    st.stop()

st.divider()
st.subheader("📅 Select a Date")

selected_date = st.date_input(
    "Prediction date",
    value=forecast_dates[0],
    min_value=forecast_dates[0],
    max_value=forecast_dates[-1],
    help="Available forecast dates mein se date choose karo."
)

selected_rows = daily[daily["date"] == selected_date]

if selected_rows.empty:
    st.warning("Selected date ka API data available nahi hai.")
    st.stop()

selected_weather = selected_rows.iloc[0]

st.write(f"### Weather for {selected_date.strftime('%d %B %Y')}")

a, b, c, d = st.columns(4)

if pd.notna(selected_weather["temperature_2m_mean"]):
    a.metric(
        "Open-Meteo Mean Forecast",
        f'{selected_weather["temperature_2m_mean"]:.1f} °C'
    )
else:
    a.metric("Open-Meteo Mean Forecast", "Unavailable")

if pd.notna(selected_weather["temperature_2m_max"]):
    b.metric(
        "Forecast Maximum",
        f'{selected_weather["temperature_2m_max"]:.1f} °C'
    )

if pd.notna(selected_weather["temperature_2m_min"]):
    c.metric(
        "Forecast Minimum",
        f'{selected_weather["temperature_2m_min"]:.1f} °C'
    )

if pd.notna(selected_weather["precipitation_sum"]):
    d.metric(
        "Precipitation",
        f'{selected_weather["precipitation_sum"]:.1f} mm'
    )


# ML prediction button
if st.button("🤖 Run ML Prediction", type="primary"):
    try:
        ml_prediction = predict_for_date(
            model,
            feature_names,
            daily,
            selected_date
        )

        st.success(
            f"ML estimated mean temperature: "
            f"{ml_prediction:.1f} °C"
        )

        st.caption(
            "ML estimate tumhare trained Random Forest model se aata hai. "
            "Multiple future days ke liye predictions recursive hain, "
            "isliye door ki dates par uncertainty badh sakti hai."
        )

        if pd.notna(selected_weather["temperature_2m_mean"]):
            st.info(
                "Open-Meteo result aur ML estimate alag models/methods "
                "se aate hain. Inka difference automatically ML error "
                "nahi maana ja sakta."
            )

    except Exception as error:
        st.error(f"ML prediction nahi ho paayi: {error}")


# Show forecast table
st.divider()
st.subheader("📊 Available Daily Forecast")

display_columns = [
    "date",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max"
]

forecast_table = daily[
    daily["date"].isin(forecast_dates)
][display_columns].copy()

forecast_table.columns = [
    "Date",
    "Mean Temp (°C)",
    "Max Temp (°C)",
    "Min Temp (°C)",
    "Precipitation (mm)",
    "Max Wind (km/h)"
]

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)

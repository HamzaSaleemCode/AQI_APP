"""
Faisalabad AQI Monitoring & Forecasting Dashboard
---------------------------------------------------
A Streamlit front-end for the AQI data-cleaning / EDA / ML-forecasting
pipeline built in the companion notebook (AQI_FSD_Pipeline.ipynb).

Run with:
    streamlit run app.py

Expects FSD_AQI_dummy_20k.csv to sit next to this file (or upload one
from the sidebar).
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════
DATA_PATH = "FSD_AQI_dummy_20k.csv"
RANDOM_SEED = 42

FEATURE_COLS = [
    "pm25_ugm3", "pm10_ugm3", "co_mgm3", "no2_ugm3",
    "no_ugm3", "nox_ugm3", "so2_ugm3", "o3_ugm3",
    "temperature_c", "humidity_pct", "wind_speed_ms",
    "wind_direction_deg", "pressure_hpa", "solar_radiation_wm2",
    "rainfall_mmhr",
]

POLLUTANT_COLS = ["pm25_ugm3", "pm10_ugm3", "co_mgm3", "no2_ugm3", "so2_ugm3", "o3_ugm3", "aqi"]

CAT_ORDER = [
    "Good", "Moderate", "Unhealthy for Sensitive Groups",
    "Unhealthy", "Very Unhealthy", "Hazardous",
]

CAT_COLORS = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
}


def aqi_to_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING  (mirrors the notebook's cleaning pipeline)
# ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading & cleaning sensor data...")
def load_and_clean(source):
    df_raw = pd.read_csv(source)
    df = df_raw.copy()

    # 1. Keep only "Online" sensor readings
    df = df[df["sensor_status"] == "Online"].copy()

    # 2. Drop duplicate timestamps per sensor
    df = df.drop_duplicates(subset=["sensor_id", "timestamp"])

    # 3. IQR x3 outlier capping per pollutant column
    for col in POLLUTANT_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        df[col] = df[col].clip(lower, upper)

    # 4. Sort + interpolate short gaps (max 3h) per sensor
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df[numeric_cols] = df.groupby("sensor_id")[numeric_cols].transform(
        lambda x: x.interpolate(method="linear", limit=3)
    )
    df = df.dropna(subset=["aqi"]).reset_index(drop=True)

    # Derived time fields used across the EDA tab
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_name"] = df["timestamp"].dt.day_name()
    df["year_month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    })
    df["aqi_category"] = df["aqi"].apply(aqi_to_category)

    return df_raw, df


# ════════════════════════════════════════════════════════════════════════
# MODELING — point-in-time AQI prediction (3-model comparison)
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training comparison models (Linear / Ridge / XGBoost)...")
def train_models(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["aqi"].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_sc, X_test_sc = scaler.fit_transform(X_train), scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
        ),
    }

    results, predictions, fitted = {}, {}, {}
    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
        preds = np.clip(preds, 0, 500)

        results[name] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
            "R2": r2_score(y_test, preds),
        }
        predictions[name] = preds
        fitted[name] = model

    best_name = min(results, key=lambda n: results[n]["RMSE"])
    return {
        "results": results, "predictions": predictions, "fitted": fitted,
        "scaler": scaler, "y_test": y_test, "best_name": best_name,
    }


# ════════════════════════════════════════════════════════════════════════
# MODELING — N-hour-ahead forecast (multi-output XGBoost, per sensor)
# ════════════════════════════════════════════════════════════════════════
def build_multistep_dataset(df: pd.DataFrame, horizon: int):
    """Builds (X, y) where y has `horizon` columns, one per hour ahead.
    Built per-sensor so targets never leak across different sensors."""
    X_parts, y_parts = [], []
    for _, g in df.groupby("sensor_id"):
        g = g.sort_values("timestamp")
        Xg, yg = g[FEATURE_COLS].values, g["aqi"].values
        n = len(g)
        if n <= horizon:
            continue
        targets = np.column_stack([yg[h:n - horizon + h] for h in range(1, horizon + 1)])
        X_parts.append(Xg[: n - horizon])
        y_parts.append(targets)
    return np.vstack(X_parts), np.vstack(y_parts)


@st.cache_resource(show_spinner="Training multi-step forecast model...")
def train_forecast_model(df: pd.DataFrame, horizon: int):
    X, y = build_multistep_dataset(df, horizon=horizon)
    base = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=RANDOM_SEED, verbosity=0,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X, y)
    return model


def generate_forecast(model, df: pd.DataFrame, sensor_id: str, horizon: int):
    g = df[df["sensor_id"] == sensor_id].sort_values("timestamp")
    last_row = g.iloc[-1]
    last_ts = last_row["timestamp"]
    x_input = last_row[FEATURE_COLS].values.reshape(1, -1)

    preds = np.clip(model.predict(x_input)[0], 0, 500)
    rows = [
        {
            "forecast_timestamp": last_ts + pd.Timedelta(hours=h),
            "hours_ahead": h,
            "predicted_aqi": round(float(p), 1),
            "aqi_category": aqi_to_category(float(p)),
        }
        for h, p in enumerate(preds, start=1)
    ]
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Faisalabad AQI Dashboard", page_icon="🌫️", layout="wide")

# ── Sidebar: data source ───────────────────────────────────────────────
st.sidebar.title("🌫️ AQI Dashboard")
st.sidebar.markdown("**Faisalabad Air Quality** — FSD project")

uploaded = st.sidebar.file_uploader("Upload sensor CSV (optional)", type="csv")
source = uploaded if uploaded is not None else DATA_PATH

try:
    df_raw, df_clean = load_and_clean(source)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}` next to `app.py`. "
        "Upload a CSV from the sidebar, or place the file alongside this script."
    )
    st.stop()

# ── Sidebar: filters ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

sensor_options = sorted(df_clean["sensor_id"].unique())
selected_sensors = st.sidebar.multiselect("Sensors", sensor_options, default=sensor_options)

min_date, max_date = df_clean["timestamp"].min().date(), df_clean["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
start_date, end_date = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_date, max_date)

mask = (
    df_clean["sensor_id"].isin(selected_sensors)
    & (df_clean["timestamp"].dt.date >= start_date)
    & (df_clean["timestamp"].dt.date <= end_date)
)
fdf = df_clean[mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Raw rows: {len(df_raw):,}\n\n"
    f"Clean rows: {len(df_clean):,}\n\n"
    f"Filtered rows: {len(fdf):,}"
)

if fdf.empty:
    st.warning("No data matches the current filters — widen the sensor or date selection in the sidebar.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════
# HEADER METRICS
# ════════════════════════════════════════════════════════════════════════
st.title("🌫️ Faisalabad AQI Monitoring & Forecasting Dashboard")
st.caption("Sensor data → cleaning → EDA → ML forecasting, in one interactive view.")

latest = fdf.sort_values("timestamp").groupby("sensor_id").tail(1)
latest_mean_aqi = latest["aqi"].mean()
latest_cat = aqi_to_category(latest_mean_aqi)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Avg AQI", f"{latest_mean_aqi:.0f}")
m1.markdown(
    f"<span style='background-color:{CAT_COLORS[latest_cat]};padding:2px 10px;"
    f"border-radius:8px;font-weight:600;font-size:0.85rem;'>{latest_cat}</span>",
    unsafe_allow_html=True,
)
m2.metric("Period Mean AQI", f"{fdf['aqi'].mean():.1f}")
m3.metric("Period Max AQI", f"{fdf['aqi'].max():.0f}")
m4.metric("Sensors in View", f"{fdf['sensor_id'].nunique()}")

# ════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════
tab_overview, tab_trends, tab_models, tab_forecast, tab_data = st.tabs(
    ["📍 Overview", "📈 Trends & EDA", "🤖 Model Performance", "📅 N-Hour Forecast", "🗂️ Data Explorer"]
)

# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────
with tab_overview:
    col_map, col_dist = st.columns([1.3, 1])

    with col_map:
        st.subheader("Sensor Locations (sized & colored by latest AQI)")
        loc_df = latest[["sensor_id", "location", "latitude", "longitude", "aqi", "aqi_category"]]
        fig_map = px.scatter_mapbox(
            loc_df, lat="latitude", lon="longitude",
            color="aqi_category", size="aqi",
            color_discrete_map=CAT_COLORS,
            category_orders={"aqi_category": CAT_ORDER},
            hover_name="location",
            hover_data={"sensor_id": True, "aqi": ":.0f", "latitude": False, "longitude": False},
            zoom=11, height=420,
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, width='stretch')

    with col_dist:
        st.subheader("AQI Category Mix")
        cat_counts = fdf["aqi_category"].value_counts().reindex(CAT_ORDER).dropna()
        fig_cat = px.bar(
            x=cat_counts.values, y=cat_counts.index, orientation="h",
            color=cat_counts.index, color_discrete_map=CAT_COLORS,
            labels={"x": "Hours", "y": ""},
        )
        fig_cat.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig_cat, width='stretch')

    st.subheader("Latest Reading per Sensor")
    st.dataframe(
        latest[["sensor_id", "location", "timestamp", "aqi", "aqi_category",
                "pm25_ugm3", "pm10_ugm3", "temperature_c", "humidity_pct"]]
        .sort_values("aqi", ascending=False),
        width='stretch', hide_index=True,
    )

# ── TAB 2: TRENDS & EDA ───────────────────────────────────────────────────
with tab_trends:
    st.subheader("Monthly AQI Trend")
    monthly = fdf.groupby("year_month")["aqi"].agg(["mean", "min", "max"]).reset_index()
    fig_month = go.Figure()
    fig_month.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["max"],
                                    line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_month.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["min"], fill="tonexty",
                                    fillcolor="rgba(70,130,180,0.2)", line=dict(width=0), name="Min–Max range"))
    fig_month.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["mean"],
                                    line=dict(color="steelblue", width=3), mode="lines+markers", name="Mean AQI"))
    fig_month.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy (150)")
    fig_month.add_hline(y=100, line_dash="dot", line_color="orange", annotation_text="Moderate (100)")
    fig_month.update_layout(height=380, xaxis_title="Month", yaxis_title="AQI")
    st.plotly_chart(fig_month, width='stretch')

    col_h, col_d, col_s = st.columns(3)
    with col_h:
        st.markdown("**Mean AQI by Hour of Day**")
        hourly = fdf.groupby("hour")["aqi"].mean().reset_index()
        st.plotly_chart(
            px.bar(hourly, x="hour", y="aqi", color_discrete_sequence=["steelblue"]),
            width='stretch',
        )
    with col_d:
        st.markdown("**Mean AQI by Day of Week**")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = fdf.groupby("day_name")["aqi"].mean().reindex(day_order).reset_index()
        st.plotly_chart(
            px.bar(dow, x="day_name", y="aqi", color_discrete_sequence=["coral"]),
            width='stretch',
        )
    with col_s:
        st.markdown("**AQI Distribution by Season**")
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        st.plotly_chart(
            px.box(fdf, x="season", y="aqi", category_orders={"season": season_order}, color="season"),
            width='stretch',
        )

    st.subheader("Pollutant & Meteorological Correlation")
    corr_cols = ["pm25_ugm3", "pm10_ugm3", "co_mgm3", "no2_ugm3", "so2_ugm3", "o3_ugm3",
                 "temperature_c", "humidity_pct", "wind_speed_ms", "rainfall_mmhr", "aqi"]
    corr = fdf[corr_cols].corr()
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn_r", zmin=-1, zmax=1, aspect="auto"),
        width='stretch',
    )

    st.subheader("Top 10 Worst AQI Hours (in current selection)")
    top10 = fdf.nlargest(10, "aqi")[
        ["timestamp", "sensor_id", "location", "aqi", "aqi_category", "pm25_ugm3", "pm10_ugm3"]
    ]
    st.dataframe(top10, width='stretch', hide_index=True)

# ── TAB 3: MODEL PERFORMANCE ─────────────────────────────────────────────
with tab_models:
    st.subheader("Model Training & Comparison")
    st.caption(
        "Trained once on the full cleaned dataset (chronological 80/20 split) — "
        "independent of the sidebar filters, so results stay stable while you explore."
    )

    bundle = train_models(df_clean)
    results_df = pd.DataFrame(bundle["results"]).T
    results_df.columns = ["MAE", "RMSE", "R2"]
    results_df = results_df.sort_values("RMSE")

    st.dataframe(
        results_df.style
        .highlight_min(subset=["MAE", "RMSE"], color="#d4f4dd")
        .highlight_max(subset=["R2"], color="#d4f4dd")
        .format("{:.3f}"),
        width='stretch',
    )
    st.success(f"Best model on this dataset: **{bundle['best_name']}**")

    model_names = list(bundle["results"].keys())
    model_choice = st.selectbox("Inspect a model", model_names, index=model_names.index(bundle["best_name"]))
    preds, y_test = bundle["predictions"][model_choice], bundle["y_test"]
    n = min(500, len(y_test))

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=y_test[-n:], name="Actual", line=dict(color="steelblue")))
    fig_pred.add_trace(go.Scatter(y=preds[-n:], name="Predicted", line=dict(color="coral", dash="dash")))
    fig_pred.update_layout(
        title=f"Actual vs Predicted — {model_choice} (last {n} test points)",
        xaxis_title="Test sample index", yaxis_title="AQI", height=380,
    )
    st.plotly_chart(fig_pred, width='stretch')

    col_imp, col_res = st.columns(2)
    with col_imp:
        if model_choice == "XGBoost":
            fi = (
                pd.Series(bundle["fitted"]["XGBoost"].feature_importances_, index=FEATURE_COLS)
                .sort_values(ascending=False)
                .head(10)
            )
            st.plotly_chart(
                px.bar(fi, orientation="h", title="Top 10 Feature Importances (XGBoost)",
                       labels={"value": "Importance", "index": ""}),
                width='stretch',
            )
        else:
            st.info("Feature importance is only available for the XGBoost model — select it above to view.")
    with col_res:
        residuals = y_test - preds
        st.plotly_chart(
            px.histogram(residuals, nbins=60, title="Residual Distribution",
                         labels={"value": "Actual − Predicted"}),
            width='stretch',
        )

# ── TAB 4: N-HOUR FORECAST ───────────────────────────────────────────────
with tab_forecast:
    st.subheader("Multi-Step AQI Forecast")
    st.caption("A single XGBoost-based model predicts every hour of the horizon at once from the sensor's latest reading.")

    col_a, col_b = st.columns(2)
    sel_sensor = col_a.selectbox("Sensor", sensor_options)
    horizon = col_b.slider("Forecast horizon (hours)", min_value=6, max_value=48, value=24, step=6)

    fmodel = train_forecast_model(df_clean, horizon=horizon)
    fcst = generate_forecast(fmodel, df_clean, sel_sensor, horizon=horizon)

    fig_fc = go.Figure()
    bands = [
        (0, 50, "#00E400"), (51, 100, "#FFFF00"), (101, 150, "#FF7E00"),
        (151, 200, "#FF0000"), (201, 300, "#8F3F97"), (301, 500, "#7E0023"),
    ]
    for lo, hi, color in bands:
        fig_fc.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.12, line_width=0)
    fig_fc.add_trace(go.Scatter(
        x=fcst["hours_ahead"], y=fcst["predicted_aqi"],
        mode="lines+markers+text",
        text=fcst["predicted_aqi"].round(0).astype(int),
        textposition="top center",
        line=dict(color="navy", width=3),
        name="Predicted AQI",
    ))
    fig_fc.update_layout(
        height=420, xaxis_title="Hours ahead", yaxis_title="Predicted AQI",
        title=f"{horizon}-Hour Forecast — {sel_sensor}",
        yaxis_range=[0, max(350, fcst["predicted_aqi"].max() + 50)],
    )
    st.plotly_chart(fig_fc, width='stretch')

    st.dataframe(fcst, width='stretch', hide_index=True)

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        "⬇️ Download forecast CSV", fcst.to_csv(index=False),
        file_name=f"forecast_{sel_sensor}_{horizon}h.csv", mime="text/csv",
    )
    col_dl2.download_button(
        "⬇️ Download forecast JSON", fcst.to_json(orient="records", date_format="iso", indent=2),
        file_name=f"forecast_{sel_sensor}_{horizon}h.json", mime="application/json",
    )

# ── TAB 5: DATA EXPLORER ─────────────────────────────────────────────────
with tab_data:
    st.subheader("Filtered Clean Data")
    st.dataframe(fdf, width='stretch', hide_index=True)
    st.download_button(
        "⬇️ Download filtered data as CSV", fdf.to_csv(index=False),
        file_name="aqi_filtered.csv", mime="text/csv",
    )

    st.subheader("Summary Statistics")
    st.dataframe(fdf.describe().T, width='stretch')

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "Gold_Price_Final_ModelData_External_Return.csv"
COMPARISON_DIR = APP_DIR / "comparison"
METRICS_PATH = COMPARISON_DIR / "H1_Model_Comparison_Metrics.csv"
RANKING_PATH = COMPARISON_DIR / "H1_Model_Ranking.csv"
PREDICTIONS_PATH = COMPARISON_DIR / "H1_Combined_WalkForward_Predictions.csv"
COMPARISON_CONFIG_PATH = COMPARISON_DIR / "H1_Comparison_Configuration.json"

MODEL_ORDER = ["MLR", "SVR", "KNN", "RF"]
MODEL_PATHS = {
    "MLR": APP_DIR / "models" / "MLR" / "MLR_Deployment_Model.joblib",
    "SVR": APP_DIR / "models" / "SVR" / "SVR_Deployment_Pipeline.joblib",
    "KNN": APP_DIR / "models" / "KNN" / "KNN_Deployment_Pipeline.joblib",
    "RF": APP_DIR / "models" / "RF" / "RandomForest_Deployment_Model.joblib",
}
CONFIG_PATHS = {
    "MLR": APP_DIR / "models" / "MLR" / "MLR_Configuration.json",
    "SVR": APP_DIR / "models" / "SVR" / "SVR_Configuration.json",
    "KNN": APP_DIR / "models" / "KNN" / "KNN_Configuration.json",
    "RF": APP_DIR / "models" / "RF" / "RandomForest_Configuration.json",
}

EXPECTED_CANONICAL_SHA256 = "bbfaf218b9784999210a06e92587ca2135d69f4d7d874e68b16dd2045dd37e64"
EXPECTED_SKLEARN_VERSION = "1.6.1"
EXPECTED_JOBLIB_VERSION = "1.5.3"

PREDICTORS = [
    "Current_Price",
    "Current_Open",
    "Current_High",
    "Current_Low",
    "Current_Volume",
    "Price_Lag1",
    "Price_Lag2",
    "Current_CHG",
    "MA_7",
    "MA_30",
    "Volatility_7",
    "Volatility_30",
    "Momentum_7",
    "Momentum_30",
    "USD_Index_Return_Lag1",
    "US10Y_Real_Yield_Change_Lag1",
]

MANUAL_PREDICTORS = [
    "Current_Price",
    "Current_Open",
    "Current_High",
    "Current_Low",
    "Current_Volume",
    "Price_Lag1",
    "Price_Lag2",
]

CALCULATED_PREDICTORS = [
    "Current_CHG",
    "MA_7",
    "MA_30",
    "Volatility_7",
    "Volatility_30",
    "Momentum_7",
    "Momentum_30",
]

EXTERNAL_PREDICTORS = [
    "USD_Index_Return_Lag1",
    "US10Y_Real_Yield_Change_Lag1",
]

TARGET_FIELDS = {"Target_Next_Return", "Target_Next_Price", "Target_Date", "Split"}

INPUT_GROUPS = {
    "Current Gold Market Data": [
        "Current_Price", "Current_Open", "Current_High", "Current_Low",
        "Current_Volume", "Current_CHG",
    ],
    "Historical and Technical Inputs": [
        "Price_Lag1", "Price_Lag2", "MA_7", "MA_30",
        "Volatility_7", "Volatility_30", "Momentum_7", "Momentum_30",
    ],
    "External Economic Inputs": [
        "USD_Index_Return_Lag1", "US10Y_Real_Yield_Change_Lag1",
    ],
}

FIELD_LABELS = {
    "Current_Price": "Current Price",
    "Current_Open": "Current Open",
    "Current_High": "Current High",
    "Current_Low": "Current Low",
    "Current_Volume": "Current Volume",
    "Current_CHG": "Current CHG",
    "Price_Lag1": "Price Lag 1",
    "Price_Lag2": "Price Lag 2",
    "MA_7": "MA 7",
    "MA_30": "MA 30",
    "Volatility_7": "Volatility 7",
    "Volatility_30": "Volatility 30",
    "Momentum_7": "Momentum 7",
    "Momentum_30": "Momentum 30",
    "USD_Index_Return_Lag1": "USD Index Return Lag 1",
    "US10Y_Real_Yield_Change_Lag1": "US 10-Year Real Yield Change Lag 1",
}

FIELD_HELP = {
    "Current_Price": "Gold price at the latest completed observation.",
    "Current_Open": "Opening gold price for the latest completed observation.",
    "Current_High": "Highest gold price for the latest completed observation.",
    "Current_Low": "Lowest gold price for the latest completed observation.",
    "Current_Volume": "Recorded market volume for the latest completed observation.",
    "Current_CHG": "Current recorded percentage change, expressed as a decimal.",
    "Price_Lag1": "Gold price from the previous recorded observation.",
    "Price_Lag2": "Gold price from two recorded observations earlier.",
    "MA_7": "Seven-observation moving average of gold price.",
    "MA_30": "Thirty-observation moving average of gold price.",
    "Volatility_7": "Seven-observation return volatility, expressed as a decimal.",
    "Volatility_30": "Thirty-observation return volatility, expressed as a decimal.",
    "Momentum_7": "Seven-observation momentum, expressed as a decimal.",
    "Momentum_30": "Thirty-observation momentum, expressed as a decimal.",
    "USD_Index_Return_Lag1": "Previous recorded U.S. Dollar Index return.",
    "US10Y_Real_Yield_Change_Lag1": "Previous recorded change in the 10-year U.S. real yield.",
}

MODEL_COLORS = {
    "MLR": "#B08D3E",
    "SVR": "#2F6F8F",
    "KNN": "#2A9D8F",
    "RF": "#C65D4B",
    "Persistence": "#6B7280",
    "Actual": "#172B4D",
}


class AppValidationError(RuntimeError):
    """Raised when a bundled artifact violates the frozen H1 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def load_csv(path_text: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    return pd.read_csv(path_text, parse_dates=list(parse_dates) or None)


@st.cache_data(show_spinner=False)
def load_json(path_text: str) -> dict[str, Any]:
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading frozen deployment models…")
def load_models() -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for label, path in MODEL_PATHS.items():
        try:
            artifact = joblib.load(path)
            if label == "MLR" and isinstance(artifact, dict):
                if artifact.get("predictors") != PREDICTORS:
                    raise AppValidationError("MLR joblib predictor metadata is inconsistent.")
                if artifact.get("target") != "Target_Next_Return":
                    raise AppValidationError("MLR joblib target metadata is inconsistent.")
                artifact = artifact.get("model")
            if artifact is None or not callable(getattr(artifact, "predict", None)):
                raise AppValidationError(f"The bundled {label} artifact has no usable estimator.")
            loaded[label] = artifact
        except Exception as exc:
            if isinstance(exc, AppValidationError):
                raise
            raise AppValidationError(
                f"Unable to load the bundled {label} deployment model."
            ) from exc
    return loaded


def predictors_from_config(config: dict[str, Any], label: str) -> list[str]:
    for key in ("Predictors", "predictors", "predictors_in_order"):
        if key in config:
            value = config[key]
            if not isinstance(value, list):
                raise AppValidationError(f"{label} predictor metadata is not a list.")
            return value
    raise AppValidationError(f"{label} configuration has no recognized predictor field.")


def ensure_finite_predictor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape != (1, len(PREDICTORS)):
        raise ValueError("Input must contain exactly one row and all 16 predictors.")
    if list(frame.columns) != PREDICTORS:
        raise ValueError("Predictor columns must match the required names and order exactly.")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Every predictor must contain one numeric value; missing values are not allowed.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Infinite predictor values are not allowed.")
    return numeric.astype(float)


def validate_startup() -> dict[str, Any]:
    required_paths = [
        DATA_PATH, METRICS_PATH, RANKING_PATH, PREDICTIONS_PATH,
        COMPARISON_CONFIG_PATH, *MODEL_PATHS.values(), *CONFIG_PATHS.values(),
    ]
    missing = [path.relative_to(APP_DIR).as_posix() for path in required_paths if not path.is_file()]
    if missing:
        raise AppValidationError("Missing required bundled files: " + ", ".join(missing))

    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        raise AppValidationError(
            f"scikit-learn {EXPECTED_SKLEARN_VERSION} is required; found {sklearn.__version__}. "
            "Install the pinned requirements before running the app."
        )
    if joblib.__version__ != EXPECTED_JOBLIB_VERSION:
        raise AppValidationError(
            f"joblib {EXPECTED_JOBLIB_VERSION} is required; found {joblib.__version__}."
        )

    canonical_hash = sha256_file(DATA_PATH)
    if canonical_hash != EXPECTED_CANONICAL_SHA256:
        raise AppValidationError("Canonical dataset SHA-256 does not match the frozen value.")

    canonical = load_csv(str(DATA_PATH))
    if canonical.shape != (3073, 21):
        raise AppValidationError(f"Canonical dataset must be 3073 × 21; found {canonical.shape}.")
    if list(canonical.columns[1:17]) != PREDICTORS:
        raise AppValidationError("Canonical predictor names or order do not match the frozen contract.")
    canonical_predictors = canonical[PREDICTORS].apply(pd.to_numeric, errors="coerce")
    if canonical_predictors.isna().any().any():
        raise AppValidationError("Canonical predictors contain missing or non-numeric values.")
    if not np.isfinite(canonical_predictors.to_numpy(dtype=float)).all():
        raise AppValidationError("Canonical predictors contain infinite values.")
    if TARGET_FIELDS.intersection(PREDICTORS):
        raise AppValidationError("A target or split field has entered the predictor contract.")

    configurations = {
        label: load_json(str(path)) for label, path in CONFIG_PATHS.items()
    }
    for label, config in configurations.items():
        if predictors_from_config(config, label) != PREDICTORS:
            raise AppValidationError(f"{label} configuration predictor order is inconsistent.")
        configured_target = config.get("Target", config.get("target"))
        if configured_target != "Target_Next_Return":
            raise AppValidationError(f"{label} configuration target is inconsistent.")

    metrics = load_csv(str(METRICS_PATH))
    ranking = load_csv(str(RANKING_PATH))
    historical = load_csv(
        str(PREDICTIONS_PATH), parse_dates=("Origin_Date", "Target_Date")
    )
    if metrics.shape != (5, 12):
        raise AppValidationError(f"Comparison metrics must be 5 × 12; found {metrics.shape}.")
    if set(ranking["Model"]) != set(MODEL_ORDER) or ranking.shape[0] != 4:
        raise AppValidationError("Ranking must contain MLR, SVR, KNN and RF exactly once.")
    if historical.shape != (2460, 16):
        raise AppValidationError(
            f"Historical prediction data must be 2460 × 16; found {historical.shape}."
        )
    expected_counts = {label: 615 for label in MODEL_ORDER}
    if historical.groupby("Model").size().to_dict() != expected_counts:
        raise AppValidationError("Historical predictions must contain 615 rows per model.")

    alignment_columns = [
        "Evaluation_Step", "Origin_Date", "Target_Date", "Current_Price",
        "Actual_Next_Return", "Actual_Next_Price",
    ]
    reference = (
        historical.loc[historical["Model"].eq("MLR"), alignment_columns]
        .sort_values("Evaluation_Step")
        .reset_index(drop=True)
    )
    for label in MODEL_ORDER[1:]:
        candidate = (
            historical.loc[historical["Model"].eq(label), alignment_columns]
            .sort_values("Evaluation_Step")
            .reset_index(drop=True)
        )
        if not candidate.equals(reference):
            raise AppValidationError(f"Historical dates or outcomes are misaligned for {label}.")

    models = load_models()
    for label, model in models.items():
        feature_count = getattr(model, "n_features_in_", None)
        if feature_count is not None and int(feature_count) != len(PREDICTORS):
            raise AppValidationError(f"{label} deployment model does not expect 16 predictors.")
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None and list(feature_names) != PREDICTORS:
            raise AppValidationError(f"{label} deployment model feature order is inconsistent.")

    for label in ("SVR", "KNN"):
        pipeline = models[label]
        if not isinstance(pipeline, Pipeline):
            raise AppValidationError(f"{label} deployment artifact must be a scikit-learn Pipeline.")
        if not any(isinstance(step, StandardScaler) for _, step in pipeline.steps):
            raise AppValidationError(f"{label} Pipeline does not contain its required StandardScaler.")

    test_frame = ensure_finite_predictor_frame(canonical.iloc[[-1]][PREDICTORS].copy())
    smoke_predictions: dict[str, float] = {}
    for label, model in models.items():
        value = float(np.asarray(model.predict(test_frame)).reshape(-1)[0])
        if not np.isfinite(value):
            raise AppValidationError(f"{label} produced a non-finite return in startup validation.")
        reconstructed = float(test_frame.iloc[0]["Current_Price"] * (1.0 + value))
        if not np.isfinite(reconstructed):
            raise AppValidationError(f"{label} produced a non-finite reconstructed price.")
        smoke_predictions[label] = value

    comparison_config = load_json(str(COMPARISON_CONFIG_PATH))
    if comparison_config.get("model_fitting_or_tuning_performed") is not False:
        raise AppValidationError("Historical comparison metadata does not confirm frozen predictions.")
    if comparison_config.get("comparison_joblib_created") is not False:
        raise AppValidationError("Historical comparison metadata indicates an unexpected joblib.")

    return {
        "canonical": canonical,
        "metrics": metrics,
        "ranking": ranking,
        "historical": historical,
        "models": models,
        "canonical_hash": canonical_hash,
        "smoke_predictions": smoke_predictions,
    }


def direction_from_return(value: float) -> str:
    if np.isclose(value, 0.0, rtol=0.0, atol=1e-12):
        return "Flat"
    return "Up" if value > 0 else "Down"


def predict_all_models(models: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_finite_predictor_frame(frame)
    current_price = float(frame.iloc[0]["Current_Price"])
    rows: list[dict[str, Any]] = []
    for label in MODEL_ORDER:
        predicted_return = float(np.asarray(models[label].predict(frame)).reshape(-1)[0])
        predicted_price = current_price * (1.0 + predicted_return)
        if not np.isfinite(predicted_return) or not np.isfinite(predicted_price):
            raise ValueError(f"{label} produced a non-finite prediction.")
        rows.append({
            "Model": label,
            "Current Price": current_price,
            "Predicted Next Return": predicted_return,
            "Predicted Return Percentage": 100.0 * predicted_return,
            "Predicted Price Change": predicted_price - current_price,
            "Predicted Next Price": predicted_price,
            "Direction": direction_from_return(predicted_return),
        })
    return pd.DataFrame(rows)


def apply_plot_style(fig: go.Figure, *, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=62, b=32),
        paper_bgcolor="white",
        plot_bgcolor="#FBFCFE",
        font=dict(family="Arial, sans-serif", color="#334155"),
        title_font=dict(size=19, color="#172B4D"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#94A3B8")
    return fig


def render_prediction_results(
    results: pd.DataFrame, current_price: float, origin_date: object
) -> None:
    st.subheader("Four-model H1 prediction")
    st.caption(
        "Each deployment model predicts the next recorded return. The displayed price is "
        "reconstructed from the supplied Current_Price."
    )
    st.caption(f"Origin Date (record-keeping only; excluded from model input): {origin_date}")
    styled = results.style.format({
        "Current Price": "{:,.2f}",
        "Predicted Next Return": "{:+.8f}",
        "Predicted Return Percentage": "{:+.4f}%",
        "Predicted Price Change": "{:+,.2f}",
        "Predicted Next Price": "{:,.2f}",
    })
    st.dataframe(styled, width="stretch", hide_index=True)

    price_spread = float(
        results["Predicted Next Price"].max() - results["Predicted Next Price"].min()
    )
    return_spread = float(
        results["Predicted Return Percentage"].max()
        - results["Predicted Return Percentage"].min()
    )
    st.info(
        f"The four predicted next prices span {price_spread:,.2f} price units, "
        f"equivalent to a {return_spread:.4f} percentage-point spread in predicted returns. "
        "This is model disagreement, not evidence that one future prediction is already best."
    )

    if (results["Predicted Next Price"] <= 0).any():
        st.warning("At least one reconstructed price is non-positive. Recheck the supplied predictors.")

    price_fig = go.Figure()
    price_fig.add_bar(
        x=results["Model"],
        y=results["Predicted Next Price"],
        marker_color=[MODEL_COLORS[label] for label in results["Model"]],
        text=[f"{value:,.2f}" for value in results["Predicted Next Price"]],
        textposition="outside",
        hovertemplate="%{x}<br>Predicted next price: %{y:,.2f}<extra></extra>",
    )
    price_fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="#6B7280",
        annotation_text=f"Current price: {current_price:,.2f}",
        annotation_position="bottom right",
    )
    price_fig.update_layout(title="Predicted Next Recorded Gold Price", showlegend=False)
    price_fig.update_yaxes(title="Gold price", tickformat=",.0f")
    st.plotly_chart(apply_plot_style(price_fig), width="stretch", config={"displaylogo": False})

    return_fig = go.Figure()
    colors = ["#2A9D8F" if value >= 0 else "#C65D4B" for value in results["Predicted Next Return"]]
    return_fig.add_bar(
        x=results["Model"],
        y=100 * results["Predicted Next Return"],
        marker_color=colors,
        text=[f"{100 * value:+.3f}%" for value in results["Predicted Next Return"]],
        textposition="outside",
        hovertemplate="%{x}<br>Predicted next return: %{y:+.4f}%<extra></extra>",
    )
    return_fig.add_hline(y=0, line_dash="dash", line_color="#6B7280")
    return_fig.update_layout(title="Predicted Next Return", showlegend=False)
    return_fig.update_yaxes(title="Predicted return (%)", ticksuffix="%")
    st.plotly_chart(apply_plot_style(return_fig), width="stretch", config={"displaylogo": False})


def build_model_input_from_seven(
    canonical: pd.DataFrame,
    manual_values: dict[str, float],
) -> pd.DataFrame:
    """Construct the frozen 16-feature model row from seven user-entered fields."""
    values = {name: float(manual_values[name]) for name in MANUAL_PREDICTORS}

    positive_fields = [
        "Current_Price", "Current_Open", "Current_High", "Current_Low",
        "Price_Lag1", "Price_Lag2",
    ]
    if any(values[name] <= 0 for name in positive_fields):
        raise ValueError("Price, Open, High, Low and both lag prices must be greater than zero.")
    if values["Current_Volume"] < 0:
        raise ValueError("Current Volume cannot be negative.")
    if values["Current_High"] < max(
        values["Current_Open"], values["Current_Low"], values["Current_Price"]
    ):
        raise ValueError("Current High cannot be below Open, Low or Current Price.")
    if values["Current_Low"] > min(
        values["Current_Open"], values["Current_High"], values["Current_Price"]
    ):
        raise ValueError("Current Low cannot be above Open, High or Current Price.")

    stored_prices = pd.to_numeric(canonical["Current_Price"], errors="raise").astype(float)
    if len(stored_prices) < 30:
        raise ValueError("At least 30 stored historical prices are required.")

    # The seven-field interface represents one new current observation after the
    # bundled history. Older observations come from the frozen dataset, while the
    # two most recent lags and current price come from the user.
    older_prices = stored_prices.iloc[-30:-2].to_numpy(dtype=float)
    price_path = np.concatenate([
        older_prices,
        np.array([
            values["Price_Lag2"],
            values["Price_Lag1"],
            values["Current_Price"],
        ], dtype=float),
    ])
    if price_path.size != 31 or not np.isfinite(price_path).all():
        raise ValueError("Unable to construct the required 31-observation price history.")

    returns = pd.Series(price_path, dtype=float).pct_change().dropna()
    row: dict[str, float] = dict(values)
    row["Current_CHG"] = float(price_path[-1] / price_path[-2] - 1.0)
    row["MA_7"] = float(np.mean(price_path[-7:]))
    row["MA_30"] = float(np.mean(price_path[-30:]))
    row["Volatility_7"] = float(returns.iloc[-7:].std(ddof=1))
    row["Volatility_30"] = float(returns.iloc[-30:].std(ddof=1))
    row["Momentum_7"] = float(price_path[-1] / price_path[-8] - 1.0)
    row["Momentum_30"] = float(price_path[-1] / price_path[-31] - 1.0)

    # These are the latest external lag values bundled with the frozen dataset.
    # They are not live market values and are disclosed as such in the interface.
    latest_stored = canonical.iloc[-1]
    for name in EXTERNAL_PREDICTORS:
        row[name] = float(latest_stored[name])

    frame = pd.DataFrame([[row[name] for name in PREDICTORS]], columns=PREDICTORS)
    return ensure_finite_predictor_frame(frame)


def render_manual_input(bundle: dict[str, Any]) -> None:
    canonical = bundle["canonical"]
    latest_row = canonical.iloc[-1]
    previous_row = canonical.iloc[-2]
    latest_origin_date = pd.to_datetime(latest_row["Origin_Date"]).date()

    # Defaults represent a convenient editable starting point for a new record.
    defaults = {
        "Current_Price": float(latest_row["Current_Price"]),
        "Current_Open": float(latest_row["Current_Open"]),
        "Current_High": float(latest_row["Current_High"]),
        "Current_Low": float(latest_row["Current_Low"]),
        "Current_Volume": float(latest_row["Current_Volume"]),
        "Price_Lag1": float(latest_row["Current_Price"]),
        "Price_Lag2": float(previous_row["Current_Price"]),
    }

    st.info(
        "Enter seven known Gold-market values. The app constructs the seven technical "
        "predictors from the bundled historical price series and uses the latest stored "
        "external lag values. No target value is requested."
    )
    st.warning(
        "The USD Index and U.S. 10-year real-yield inputs are the latest values stored in "
        "the coursework dataset; they are not fetched live. This interface is therefore "
        "an educational next-record prototype."
    )

    if st.button("Reset seven fields", type="secondary"):
        for predictor in MANUAL_PREDICTORS:
            st.session_state[f"manual_{predictor}"] = defaults[predictor]
        st.session_state["manual_origin_date"] = latest_origin_date

    for predictor in MANUAL_PREDICTORS:
        st.session_state.setdefault(f"manual_{predictor}", defaults[predictor])
    st.session_state.setdefault("manual_origin_date", latest_origin_date)

    with st.form("manual_prediction_form"):
        st.date_input(
            "Origin Date",
            key="manual_origin_date",
            help="Display and record-keeping only. This date is never passed to a model.",
        )
        st.markdown("#### Known Gold Market Data")
        columns = st.columns(2)
        for index, predictor in enumerate(MANUAL_PREDICTORS):
            with columns[index % 2]:
                number_format = "%.2f" if predictor != "Current_Volume" else "%.4f"
                st.number_input(
                    FIELD_LABELS[predictor],
                    key=f"manual_{predictor}",
                    format=number_format,
                    help=f"{FIELD_HELP[predictor]} Model field: {predictor}",
                )
        submitted = st.form_submit_button("Predict Next Gold Price", type="primary")

    if submitted:
        try:
            manual_values = {
                name: st.session_state[f"manual_{name}"] for name in MANUAL_PREDICTORS
            }
            frame = build_model_input_from_seven(canonical, manual_values)

            with st.expander("Automatically constructed 16-model-input row", expanded=False):
                display_inputs = pd.DataFrame({
                    "Predictor": PREDICTORS,
                    "Value": [float(frame.iloc[0][name]) for name in PREDICTORS],
                    "Source": [
                        "User input" if name in MANUAL_PREDICTORS
                        else "Calculated from stored price history" if name in CALCULATED_PREDICTORS
                        else "Latest stored external lag (not live)"
                        for name in PREDICTORS
                    ],
                })
                st.dataframe(display_inputs, width="stretch", hide_index=True)

            results = predict_all_models(bundle["models"], frame)
            render_prediction_results(
                results,
                float(frame.iloc[0]["Current_Price"]),
                st.session_state["manual_origin_date"],
            )
        except ValueError as exc:
            st.error(str(exc))


def render_future_prediction(bundle: dict[str, Any]) -> None:
    st.header("H1 Future Prediction")
    st.write(
        "Supply information available at the latest completed observation. The four frozen "
        "deployment models predict **Target_Next_Return**, which is converted into a next "
        "recorded price using the current price."
    )
    st.warning(
        "The next recorded observation is not a guaranteed calendar-day forecast. Do not enter "
        "Target_Next_Return, Target_Next_Price, Target_Date or Split."
    )
    st.caption(
        "Enter seven known Gold-market fields. The app constructs the remaining model inputs "
        "without requesting the future target."
    )
    render_manual_input(bundle)


def render_ranking_and_metrics(metrics: pd.DataFrame, ranking: pd.DataFrame) -> None:
    st.subheader("Final historical H1 ranking")
    ranking_columns = [
        "Overall_Rank", "Model", "Price_RMSE", "Price_MAE",
        "Price_MAPE_Percent", "Price_NRMSE_Percent", "Price_R2", "Return_R2",
        "Directional_Accuracy_Percent", "RMSE_Skill_vs_Persistence",
        "MAE_Skill_vs_Persistence",
    ]
    styled_ranking = ranking[ranking_columns].style.format({
        "Overall_Rank": "{:.0f}",
        "Price_RMSE": "{:,.2f}",
        "Price_MAE": "{:,.2f}",
        "Price_MAPE_Percent": "{:.3f}%",
        "Price_NRMSE_Percent": "{:.3f}%",
        "Price_R2": "{:.6f}",
        "Return_R2": "{:.6f}",
        "Directional_Accuracy_Percent": "{:.2f}%",
        "RMSE_Skill_vs_Persistence": "{:+.3%}",
        "MAE_Skill_vs_Persistence": "{:+.3%}",
    })
    st.dataframe(styled_ranking, width="stretch", hide_index=True)

    st.subheader("All-model metrics, including persistence")
    styled_metrics = metrics.style.format({
        "Price_RMSE": "{:,.2f}",
        "Price_MAE": "{:,.2f}",
        "Price_R2": "{:.6f}",
        "Price_MAPE_Percent": "{:.3f}%",
        "Price_NRMSE_Percent": "{:.3f}%",
        "Return_RMSE": "{:.6f}",
        "Return_MAE": "{:.6f}",
        "Return_R2": "{:.6f}",
        "Directional_Accuracy_Percent": "{:.2f}%",
        "RMSE_Skill_vs_Persistence": "{:+.3%}",
        "MAE_Skill_vs_Persistence": "{:+.3%}",
    }, na_rep="—")
    st.dataframe(styled_metrics, width="stretch", hide_index=True)


def render_historical_charts(
    filtered: pd.DataFrame,
    historical: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_models: list[str],
) -> None:
    reference_model = selected_models[0]
    actual = (
        filtered.loc[filtered["Model"].eq(reference_model)]
        .sort_values("Target_Date")
    )

    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=actual["Target_Date"], y=actual["Actual_Next_Price"],
        mode="lines", name="Actual next price",
        line=dict(color=MODEL_COLORS["Actual"], width=2.8),
        hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:,.2f}<extra></extra>",
    ))
    for label in selected_models:
        model_rows = filtered.loc[filtered["Model"].eq(label)].sort_values("Target_Date")
        price_fig.add_trace(go.Scatter(
            x=model_rows["Target_Date"], y=model_rows["Predicted_Next_Price"],
            mode="lines", name=label,
            line=dict(color=MODEL_COLORS[label], width=1.4),
            hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>Predicted: %{{y:,.2f}}<extra></extra>",
        ))
    price_fig.update_layout(title="Actual versus Walk-Forward Predicted Next Price")
    price_fig.update_xaxes(title="Target date")
    price_fig.update_yaxes(title="Gold price", tickformat=",.0f")
    st.plotly_chart(apply_plot_style(price_fig, height=520), width="stretch", config={"displaylogo": False})

    metric_order = selected_models + ["Persistence"]
    metric_view = metrics.set_index("Model").loc[metric_order]
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        rmse_fig = go.Figure(go.Bar(
            x=metric_order,
            y=metric_view["Price_RMSE"],
            marker_color=[MODEL_COLORS[label] for label in metric_order],
            text=[f"{value:,.1f}" for value in metric_view["Price_RMSE"]],
            textposition="outside",
            hovertemplate="%{x}<br>RMSE: %{y:,.2f}<extra></extra>",
        ))
        rmse_fig.update_layout(title="Price RMSE Comparison", showlegend=False)
        rmse_fig.update_yaxes(title="Price RMSE", tickformat=",.0f")
        st.plotly_chart(apply_plot_style(rmse_fig), width="stretch", config={"displaylogo": False})
    with metric_col2:
        mae_fig = go.Figure(go.Bar(
            x=metric_order,
            y=metric_view["Price_MAE"],
            marker_color=[MODEL_COLORS[label] for label in metric_order],
            text=[f"{value:,.1f}" for value in metric_view["Price_MAE"]],
            textposition="outside",
            hovertemplate="%{x}<br>MAE: %{y:,.2f}<extra></extra>",
        ))
        mae_fig.update_layout(title="Price MAE Comparison", showlegend=False)
        mae_fig.update_yaxes(title="Price MAE", tickformat=",.0f")
        st.plotly_chart(apply_plot_style(mae_fig), width="stretch", config={"displaylogo": False})

    normalized_fig = go.Figure()
    normalized_fig.add_bar(
        x=metric_order,
        y=metric_view["Price_MAPE_Percent"],
        name="MAPE",
        marker_color="#A8B4C2",
        text=[f"{value:.3f}%" for value in metric_view["Price_MAPE_Percent"]],
        textposition="outside",
    )
    normalized_fig.add_bar(
        x=metric_order,
        y=metric_view["Price_NRMSE_Percent"],
        name="NRMSE",
        marker_color="#2F6F8F",
        text=[f"{value:.3f}%" for value in metric_view["Price_NRMSE_Percent"]],
        textposition="outside",
    )
    normalized_fig.update_layout(title="MAPE and NRMSE Comparison", barmode="group")
    normalized_fig.update_yaxes(title="Percent", ticksuffix="%")
    st.plotly_chart(apply_plot_style(normalized_fig), width="stretch", config={"displaylogo": False})

    distribution_fig = go.Figure()
    for label in selected_models:
        model_errors = filtered.loc[filtered["Model"].eq(label), "Price_Error"]
        distribution_fig.add_trace(go.Histogram(
            x=model_errors,
            name=label,
            marker_color=MODEL_COLORS[label],
            opacity=0.58,
            nbinsx=35,
            hovertemplate=f"{label}<br>Error interval: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))
    distribution_fig.add_vline(x=0, line_dash="dash", line_color="#8B1E1E")
    distribution_fig.update_layout(
        title="Prediction Error Distribution",
        barmode="overlay",
    )
    distribution_fig.update_xaxes(title="Predicted next price − actual next price", tickformat=",.0f")
    distribution_fig.update_yaxes(title="Count")
    st.plotly_chart(apply_plot_style(distribution_fig, height=480), width="stretch", config={"displaylogo": False})

    base_actual = historical.loc[historical["Model"].eq("MLR")].sort_values("Evaluation_Step")
    always_up = 100 * (base_actual["Actual_Next_Return"] > 0).mean()
    direction_values = (
        metrics.loc[metrics["Model"].isin(selected_models)]
        .set_index("Model").loc[selected_models, "Directional_Accuracy_Percent"]
    )
    direction_fig = go.Figure()
    direction_fig.add_bar(
        x=selected_models,
        y=direction_values,
        marker_color=[MODEL_COLORS[label] for label in selected_models],
        text=[f"{value:.2f}%" for value in direction_values],
        textposition="outside",
        hovertemplate="%{x}<br>Directional accuracy: %{y:.2f}%<extra></extra>",
    )
    direction_fig.add_hline(
        y=always_up,
        line_dash="dash",
        line_color="#8B1E1E",
        annotation_text=f"Always Up: {always_up:.2f}%",
        annotation_position="bottom right",
    )
    direction_fig.update_layout(
        title="Full-Evaluation Directional Comparison versus Always-Up",
        showlegend=False,
    )
    direction_fig.update_yaxes(title="Directional accuracy (%)", range=[0, 70], ticksuffix="%")

    skill_view = metrics.loc[metrics["Model"].isin(selected_models)].set_index("Model").loc[selected_models]
    skill_fig = go.Figure()
    skill_fig.add_bar(
        x=selected_models,
        y=100 * skill_view["RMSE_Skill_vs_Persistence"],
        name="RMSE skill",
        marker_color="#2A9D8F",
        text=[f"{100 * value:+.2f}%" for value in skill_view["RMSE_Skill_vs_Persistence"]],
        textposition="outside",
    )
    skill_fig.add_bar(
        x=selected_models,
        y=100 * skill_view["MAE_Skill_vs_Persistence"],
        name="MAE skill",
        marker_color="#7A8BA3",
        text=[f"{100 * value:+.2f}%" for value in skill_view["MAE_Skill_vs_Persistence"]],
        textposition="outside",
    )
    skill_fig.add_hline(y=0, line_dash="dash", line_color="#8B1E1E")
    skill_fig.update_layout(title="Skill versus Persistence", barmode="group")
    skill_fig.update_yaxes(title="Skill (%)", ticksuffix="%")

    comparison_col1, comparison_col2 = st.columns(2)
    with comparison_col1:
        st.plotly_chart(apply_plot_style(direction_fig), width="stretch", config={"displaylogo": False})
    with comparison_col2:
        st.plotly_chart(apply_plot_style(skill_fig), width="stretch", config={"displaylogo": False})


def render_historical_comparison(bundle: dict[str, Any]) -> None:
    st.header("Historical Model Comparison")
    st.write(
        "This mode reads the frozen leakage-safe walk-forward prediction and metric CSV files. "
        "It does not call any deployment joblib."
    )
    st.info(
        "KNN ranked first descriptively. KNN and SVR only slightly beat persistence by RMSE; "
        "Return R² is negative for all four models. High Price R² is strongly influenced by "
        "adjacent-price persistence and should not be interpreted as strong return prediction."
    )
    st.caption(
        "The 615 predictions per model were generated by leakage-safe walk-forward evaluation: "
        "each target outcome remained unavailable until its prediction had been produced. "
        "These charts use the saved prediction CSV, not the final deployment joblibs. "
        "Persistence remains an essential benchmark."
    )

    metrics = bundle["metrics"].copy()
    ranking = bundle["ranking"].sort_values("Overall_Rank").copy()
    historical = bundle["historical"].copy().sort_values(["Model", "Evaluation_Step"])
    render_ranking_and_metrics(metrics, ranking)

    st.subheader("Explore saved walk-forward predictions")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_models = st.multiselect(
            "Models",
            MODEL_ORDER,
            default=MODEL_ORDER,
        )
    min_date = historical["Target_Date"].min().date()
    max_date = historical["Target_Date"].max().date()
    with col2:
        selected_dates = st.date_input(
            "Target-date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    if not selected_models:
        st.warning("Select at least one model to display the historical charts.")
        return
    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.warning("Select both a start date and an end date.")
        return
    start_date = pd.Timestamp(selected_dates[0])
    end_date = pd.Timestamp(selected_dates[1])
    if start_date > end_date:
        st.warning("The start date must not be later than the end date.")
        return

    filtered = historical.loc[
        historical["Model"].isin(selected_models)
        & historical["Target_Date"].between(start_date, end_date)
    ].copy()
    if filtered.empty:
        st.warning("No saved historical predictions match the selected filters.")
        return

    st.caption(
        f"Displaying {len(filtered):,} saved prediction rows from "
        f"{start_date.date()} to {end_date.date()}. Filtering changes only the view, not the stored results."
    )
    with st.expander("Inspect saved Evaluation predictions"):
        inspection_columns = [
            "Model", "Evaluation_Step", "Origin_Date", "Target_Date", "Current_Price",
            "Actual_Next_Return", "Predicted_Next_Return", "Actual_Next_Price",
            "Predicted_Next_Price", "Price_Error", "Correct_Direction",
        ]
        st.dataframe(
            filtered[inspection_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "Current_Price": st.column_config.NumberColumn(format="%.2f"),
                "Actual_Next_Return": st.column_config.NumberColumn(format="%.8f"),
                "Predicted_Next_Return": st.column_config.NumberColumn(format="%.8f"),
                "Actual_Next_Price": st.column_config.NumberColumn(format="%.2f"),
                "Predicted_Next_Price": st.column_config.NumberColumn(format="%.2f"),
                "Price_Error": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    render_historical_charts(filtered, historical, metrics, selected_models)


def render_sidebar(bundle: dict[str, Any]) -> str:
    with st.sidebar:
        st.markdown("## Gold Price H1")
        st.caption("Frozen four-model return-to-price prototype")
        mode = st.radio(
            "Application mode",
            ["H1 Future Prediction", "Historical Model Comparison"],
        )
        st.divider()
        st.markdown("#### Startup validation")
        st.success("All required checks passed")
        st.caption("Models loaded: 4/4")
        st.caption("Predictor contract: PASS")
        st.caption("Canonical hash: PASS")
        st.caption("Historical alignment: PASS")
        with st.expander("Model information"):
            st.write("**Target:** Target_Next_Return")
            st.write("**Predictors:** 16 ordered numeric fields")
            st.write("**Models:** MLR, SVR, KNN and Random Forest")
            st.write("SVR and KNN retain their fitted scaling steps inside saved Pipelines.")
        with st.expander("Method boundary"):
            st.write(
                "Future mode uses frozen deployment models. Historical mode uses only saved "
                "walk-forward CSV results. No model is retrained or retuned in this app."
            )
        st.divider()
        st.caption("Educational use only — not financial advice.")
    return mode


def configure_page() -> None:
    st.set_page_config(
        page_title="Gold Price H1 Model Lab",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #F7F9FC; }
        .block-container { max-width: 1420px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #172B4D; letter-spacing: -0.02em; }
        div[data-testid="stMetric"] {
            background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px;
        }
        div[data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 10px; }
        .app-kicker { color: #B08D3E; font-weight: 700; letter-spacing: .12em; font-size: .78rem; }
        .app-subtitle { color: #64748B; font-size: 1.05rem; margin-top: -0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()
    st.markdown('<div class="app-kicker">BMDS2003 · H1 PROTOTYPE</div>', unsafe_allow_html=True)
    st.title("Daily Gold Price — Next Recorded Observation")
    st.markdown(
        '<div class="app-subtitle">Predict next return, reconstruct next price, and examine frozen walk-forward evidence.</div>',
        unsafe_allow_html=True,
    )

    try:
        bundle = validate_startup()
    except AppValidationError as exc:
        st.error(f"Startup validation failed: {exc}")
        st.info("Check the bundled files and install the pinned requirements, then restart the app.")
        st.stop()

    mode = render_sidebar(bundle)
    if mode == "H1 Future Prediction":
        render_future_prediction(bundle)
    else:
        render_historical_comparison(bundle)

    st.divider()
    st.caption(
        "Educational prototype only. Model loading and validation do not guarantee accurate "
        "future financial forecasts. Gold prices can change because of factors not represented "
        "in these predictors."
    )


if __name__ == "__main__":
    main()

import os
import math
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- Config ----------------
CSV_PATH = "usd_uah.csv"
DATE_COL_CANDIDATES = ["Date", "date", "DATE", "day", "timestamp"]
PRICE_COL_CANDIDATES = ["Rate", "Close", "Price", "USD_UAH", "USD/UAH", "usd_uah", "value", "Value", "close", "price"]

FORECAST_HORIZON = 10  # days
TEST_DAYS = 10        # last N days for evaluation (if enough history)
RANDOM_STATE = 42

# -------------- Helpers -----------------
def pick_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def build_features(df, target_col):
    # Calendar features
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofmonth"] = df.index.day

    # Lag features
    for lag in [1, 2, 3, 5, 7, 14, 21]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling stats (based on shifted series to avoid leakage)
    for win in [3, 5, 7, 14, 21]:
        df[f"roll_mean_{win}"] = df[target_col].shift(1).rolling(win).mean()
        df[f"roll_std_{win}"]  = df[target_col].shift(1).rolling(win).std()

    # Drop rows with NaNs from lags/rolls
    return df.dropna().copy()

def train_model(df_feat, target_col):
    # Chronological split: last TEST_DAYS as test if enough data
    if len(df_feat) > TEST_DAYS + 30:
        split_point = df_feat.index[-TEST_DAYS]
        train = df_feat[df_feat.index < split_point]
        test  = df_feat[df_feat.index >= split_point]
    else:
        train = df_feat.copy()
        test  = df_feat.iloc[0:0].copy()

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    metrics = {"MAE": None, "RMSE": None}
    if len(test) > 0:
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]
        y_pred = model.predict(X_test)
        metrics["MAE"]  = float(mean_absolute_error(y_test, y_pred))
        metrics["RMSE"] = float(math.sqrt(mean_squared_error(y_test, y_pred)))

    return model, metrics, train, test

def recursive_forecast(df_feat, model, target_col, horizon_days):
    # df_feat has engineered features up to the last known date.
    # We'll step forward one day at a time, using predictions as new history.
    history = df_feat.copy()
    last_date = history.index[-1]
    preds = []

    # List of feature names the model expects
    feature_cols = [c for c in history.columns if c != target_col]

    for i in range(1, horizon_days + 1):
        next_date = last_date + timedelta(days=1)

        # Build a one-row DataFrame of features for next_date
        row = pd.DataFrame(index=[next_date])
        row["dayofweek"] = next_date.dayofweek
        row["month"]     = next_date.month
        row["dayofmonth"]= next_date.day

        # We need a temporary series of the target to compute lags/rollings
        tmp_target = history[target_col].copy()

        # Lags
        for lag in [1, 2, 3, 5, 7, 14, 21]:
            row[f"lag_{lag}"] = tmp_target.iloc[-lag] if len(tmp_target) >= lag else np.nan

        # Rolling stats
        for win in [3, 5, 7, 14, 21]:
            shifted = tmp_target.shift(1)
            row[f"roll_mean_{win}"] = shifted.tail(win).mean()
            row[f"roll_std_{win}"]  = shifted.tail(win).std()

        # Align to model features
        X_next = pd.DataFrame(index=[next_date], columns=feature_cols)
        for col in feature_cols:
            if col in row.columns:
                X_next.loc[next_date, col] = row.loc[next_date, col]

        X_next = X_next.apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill")

        # Predict and append to history
        yhat = float(model.predict(X_next)[0])
        preds.append((next_date, yhat))

        # Extend history with the new predicted target (for subsequent lags)
        new_hist_row = pd.DataFrame({target_col: yhat}, index=[next_date])
        history = pd.concat([history, new_hist_row], axis=0)

        last_date = next_date

    fcst = pd.DataFrame(preds, columns=["Date", "Forecast"]).set_index("Date")
    return fcst

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV не знайдено: {CSV_PATH}. Переконайтесь, що файл існує або змініть шлях у CSV_PATH.")

    # Load CSV
    df_raw = pd.read_csv(CSV_PATH)

    # Detect columns
    date_col = "Date" if "Date" in df_raw.columns else pick_column(df_raw.columns, DATE_COL_CANDIDATES)
    if date_col is None:
        raise ValueError("Не знайдено колонку з датою. Додайте 'Date' або вкажіть іншу назву з DATE_COL_CANDIDATES.")

    price_col = pick_column(df_raw.columns, PRICE_COL_CANDIDATES)
    if price_col is None:
        # If exactly 2 columns, assume non-date is price
        if len(df_raw.columns) == 2:
            price_col = [c for c in df_raw.columns if c != date_col][0]
        else:
            # fallback: last numeric column
            num_cols = df_raw.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("Не знайдено числової колонки для курсу. Додайте 'Rate' або схожу.")
            price_col = num_cols[-1]

    # Parse and sort
    df = df_raw.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # Keep only target column under a standard name
    df = df.rename(columns={price_col: "rate"})[["rate"]]
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").interpolate().bfill().ffill()

    # Build features
    df_feat = build_features(df.copy(), target_col="rate")

    # Train & evaluate
    model, metrics, train, test = train_model(df_feat.copy(), target_col="rate")

    # Forecast
    fcst = recursive_forecast(df_feat.copy(), model, target_col="rate", horizon_days=FORECAST_HORIZON)

    # Save outputs
    out_csv = "usd_uah_forecast_10days.csv"
    fcst_rounded = fcst.copy()
    fcst_rounded["Forecast"] = fcst_rounded["Forecast"].round(4)
    fcst_rounded.to_csv(out_csv, index=True)

    # Plot (single figure, no specific colors per instruction)
    merged = df.join(fcst, how="outer")
    plt.figure()
    plt.plot(merged.index, merged["rate"], label="Історичний курс")
    plt.plot(fcst.index, fcst["Forecast"], linestyle="--", marker="o", label="Прогноз (10 днів)")
    plt.title("USD/UAH: історія та 10-денний прогноз")
    plt.xlabel("Дата")
    plt.ylabel("Курс")
    plt.legend()
    plt.tight_layout()
    out_png = "usd_uah_forecast.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Print a short summary
    last_rate = float(df["rate"].iloc[-1])
    first_fc  = float(fcst["Forecast"].iloc[0])
    last_fc   = float(fcst["Forecast"].iloc[-1])
    print("--------------- РЕЗЮМЕ ---------------")
    print(f"Останній відомий курс: {last_rate:.4f}")
    print(f"Прогноз на 1-й день : {first_fc:.4f}")
    print(f"Прогноз на 10-й день: {last_fc:.4f}")
    if metrics["MAE"] is not None:
        print(f"MAE  (останні {TEST_DAYS} днів): {metrics['MAE']:.4f}")
        print(f"RMSE (останні {TEST_DAYS} днів): {metrics['RMSE']:.4f}")
    else:
        print("Недостатньо даних для об'єктивної валідації (усі дані пішли на тренування).")
    print("--------------------------------------")
    print(f"Файли збережено: {out_png}, {out_csv}")

if __name__ == "__main__":
    main()

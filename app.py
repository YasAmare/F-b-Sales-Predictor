import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("F&B Multi-Item Sales Predictor")

# --- Generate synthetic data inside the app ---
days = pd.date_range(start="2025-08-01", end="2025-12-31")
items = ["Burger", "Fries", "Drink"]

multi_sales = []
for item in items:
    for day in days:
        base = {"Burger":20, "Fries":15, "Drink":25}[item]
        weekend = 10 if day.weekday() >=5 else 0
        trend = (day - days[0]).days * 0.03
        noise = np.random.randint(-3, 4)
        multi_sales.append({
            "date": day,
            "item": item,
            "sales_qty": base + weekend + trend + noise
        })

df_multi = pd.DataFrame(multi_sales)
df_multi["day_index"] = np.arange(len(df_multi))
df_multi["day_of_week"] = df_multi["date"].dt.weekday
df_multi["is_weekend"] = (df_multi["day_of_week"] >=5).astype(int)
df_multi["trend"] = np.arange(len(df_multi)) * 0.03
df_multi = pd.get_dummies(df_multi, columns=["item"])

# --- Train model ---
feature_cols = ["day_index","day_of_week","is_weekend","trend"] + [c for c in df_multi.columns if "item_" in c]
X_multi = df_multi[feature_cols]
y_multi = df_multi["sales_qty"]
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

# --- Sidebar Inputs ---
days_ahead = st.sidebar.slider("Days to Predict", 1, 30, 7)
item_cols = [c for c in X_multi.columns if "item_" in c]

# --- Multi-day predictions ---
pred_list = []

for day_offset in range(1, days_ahead+1):
    predict_date = pd.to_datetime(df_multi["date"].iloc[-1]) + pd.Timedelta(days=day_offset)
    day_of_week = predict_date.weekday()
    is_weekend = int(day_of_week >=5)
    trend = (len(df_multi)//len(items) + day_offset -1)*0.03

    row = {"Date": predict_date.date()}
    for item_name in items:
        input_dict = {
            "day_index":[len(df_multi)//len(items) + day_offset - 1],
            "day_of_week":[day_of_week],
            "is_weekend":[is_weekend],
            "trend":[trend]
        }
        for c in item_cols:
            input_dict[c] = [1 if c==f"item_{item_name}" else 0]
        input_df = pd.DataFrame(input_dict)[X_multi.columns]
        pred = model_multi.predict(input_df)[0]
        row[item_name] = round(pred)
    pred_list.append(row)

pred_df = pd.DataFrame(pred_list)

# --- Show predictions table with alerts ---
st.subheader(f"Predicted Sales for Next {days_ahead} Days")

# Define thresholds for alerts
high_threshold = 50  # sales above this are high
low_threshold = 10   # sales below this are low

# Function to color cells
def highlight_sales(val):
    if val >= high_threshold:
        color = 'lightgreen'
    elif val <= low_threshold:
        color = 'lightcoral'
    else:
        color = ''
    return f'background-color: {color}'

# Style table with alerts
styled_pred_df = pred_df.style.applymap(highlight_sales, subset=items)
st.dataframe(styled_pred_df)

# --- Download Predictions CSV ---
st.subheader("Download Predictions")
csv = pred_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Predicted Sales as CSV",
    data=csv,
    file_name='predicted_sales.csv',
    mime='text/csv'
)

# --- Individual charts for each item ---
st.subheader("Individual Item Charts")
for item in items:
    st.write(f"**{item} Sales**")
    plt.figure(figsize=(8,3))
    col_name = f"item_{item}"
    item_df = df_multi[df_multi[col_name]==1].copy()
    plt.plot(item_df["date"], item_df["sales_qty"], label="Past Sales")
    plt.plot(pred_df["Date"], pred_df[item], label="Predicted Sales", linestyle="--")
    plt.title(f"{item} Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales Qty")
    plt.legend()
    st.pyplot(plt)

# --- Combined chart ---
st.subheader("All Items Combined Chart")
plt.figure(figsize=(8,4))
for item in items:
    col_name = f"item_{item}"
    item_df = df_multi[df_multi[col_name]==1]
    plt.plot(item_df["date"], item_df["sales_qty"], label=f"{item} Past Sales")
    plt.plot(pred_df["Date"], pred_df[item], linestyle="--", label=f"{item} Predicted")
plt.title("Multi-Item Past + Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales Quantity")
plt.legend()
st.pyplot(plt)

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
day_offset = st.sidebar.slider("Days Ahead",1,30,1)
item_name = st.sidebar.selectbox("Select Item", items)

# --- Prepare prediction input ---
predict_date = pd.to_datetime(df_multi["date"].iloc[-1]) + pd.Timedelta(days=day_offset)
day_of_week = predict_date.weekday()
is_weekend = int(day_of_week>=5)
trend = (len(df_multi)//len(items) + day_offset -1)*0.03

item_cols = [c for c in X_multi.columns if "item_" in c]
input_dict = {
    "day_index":[len(df_multi)//len(items)+day_offset-1],
    "day_of_week":[day_of_week],
    "is_weekend":[is_weekend],
    "trend":[trend]
}
for c in item_cols:
    input_dict[c] = [1 if c==f"item_{item_name}" else 0]

input_df = pd.DataFrame(input_dict)[X_multi.columns]
pred = model_multi.predict(input_df)[0]

st.write(f"Predicted sales for **{item_name}** on {predict_date.date()}: **{pred:.0f}**")

# --- Plot multi-item chart ---
plt.figure(figsize=(8,4))
for item in items:
    col_name = f"item_{item}"
    item_df = df_multi[df_multi[col_name]==1]
    plt.plot(item_df["date"], item_df["sales_qty"], label=f"{item} Sales")
plt.title("Multi-Item Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales Quantity")
plt.legend()
st.pyplot(plt)

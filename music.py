import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("spotifymusic_dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("===== DATASET =====")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==========================================
# 2. EXPLORE THE DATA
# ==========================================

plt.figure(figsize=(7, 5))

plt.scatter(
    df["Lead Streams (in millions)"],
    df["Total Streams (in millions)"]
)

plt.xlabel("Lead Streams (millions)")
plt.ylabel("Total Streams (millions)")
plt.title("Lead Streams vs Total Streams")

plt.show()


# ==========================================
# 3. CHECK FOR DATA LEAKAGE (important!)
# ==========================================
# Total Streams = Solo Streams + Collaborative Streams (exact identity)
# So Collaborative Streams (and Feature Streams, which nearly equals
# Collaborative Streams) cannot be used as predictors -- they leak
# the target. We confirm this before picking features.

leak_check = (
    df["Solo Streams (in millions)"]
    + df["Collaborative Streams (in millions)"]
    - df["Total Streams (in millions)"]
).abs()

print("\n===== LEAKAGE CHECK =====")
print("Max |Solo + Collaborative - Total| =", leak_check.max())
print("(near-zero confirms Collaborative Streams leaks the target)")


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

# Career length
df["Career Length"] = 2026 - df["Debut Year"]


# ==========================================
# 5. SELECT FEATURES AND TARGET
# ==========================================


features = [
    "Lead Streams (in millions)",
    "Debut Year"
]

X = df[features]

y = df["Total Streams (in millions)"]


print("\n===== FEATURES =====")

for feature in features:
    print(feature)

print("\n===== TARGET =====")
print("Total Streams (in millions)")


# ==========================================
# 6. SPLIT DATA
# ==========================================

Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1
)

print("\n===== DATA SPLIT =====")
print("Training rows:", len(Xtr))
print("Testing rows:", len(Xte))


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model = LinearRegression()

model.fit(Xtr, ytr)


# ==========================================
# 8. MODEL COEFFICIENTS
# ==========================================

print("\n===== MODEL =====")

print("Intercept:", model.intercept_)

print("\nCoefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")


# ==========================================
# 9. MAKE TEST PREDICTIONS
# ==========================================

pred = model.predict(Xte)


# ==========================================
# 10. EVALUATE MODEL
# ==========================================

rmse = np.sqrt(
    mean_squared_error(yte, pred)
)

r2 = r2_score(yte, pred)

print("\n===== TEST RESULTS =====")

print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")


# ==========================================
# 11. PREDICTED VS ACTUAL GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

plt.scatter(yte, pred)

# Perfect prediction line
minimum = min(yte.min(), pred.min())
maximum = max(yte.max(), pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual Total Streams")
plt.ylabel("Predicted Total Streams")
plt.title("Predicted vs Actual Total Streams ")

plt.savefig("predicted_vs_actual_fixed.png", dpi=150)
plt.show()
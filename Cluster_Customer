from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#اتصال به دیتابیس

engine = create_engine(
    "mssql+pyodbc://localhost/AdventureWorksDW2025?"
    "driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
)

# متد خواندن داده ها
def load_view(view_name):
    return pd.read_sql(
        f"SELECT * FROM {view_name}",
        engine
    )

# خواندن داده های مشتریان
clv = load_view(
    "dbo.vw_CustomerLifetimeValue"
)

# RFM خواندن
rfm = load_view(
    "dbo.vw_RFMScore"
)
df = clv.merge(
    rfm,
    on="CustomerKey",
    how="inner"
)

# انتخاب فیچرهای مهم
feature_columns = [
    "Frequency",
    "Monetary",
    "TotalRevenue",
    "TotalOrders",
    "AverageOrderValue",
    "CustomerLifetimeDays"
]

features = df[feature_columns].copy()

# حذف رکوردهای ناقص
features = features.dropna()

# هماهنگی DataFrame اصلی
df = df.loc[features.index].copy()

# استانداد سازی فیچر ها جهت صحت عملکرد مدل سازی
scaler = StandardScaler()
X = scaler.fit_transform(
    features
)

# انتخاب بهترین تعداد خوشه برای خوشه بندی
scores = []
print("\nSilhouette Scores")
print("-" * 50)
for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X)
    score = silhouette_score(
        X,
        labels
    )
    scores.append(score)
    print(
        f"K = {k}  |  Score = {score:.4f}"
    )
best_k = (
    scores.index(max(scores))
    + 2
)

print("\n" + "=" * 50)
print(f"Best K Selected: {best_k}")
print("=" * 50)

# مدل سازی نهایی با kmeans
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

# نتایج خوشه ها
cluster_summary = (
    df.groupby("Cluster")
      [
          [
              "TotalRevenue",
              "TotalOrders",
              "CustomerLifetimeDays",
              "Frequency",
              "Monetary"
          ]
      ]
      .mean()
      .round(2)
)

# محاسبه ارزش هر خوشه
cluster_summary["BusinessScore"] = (
      cluster_summary["TotalRevenue"].rank(pct=True) * 0.35
    + cluster_summary["Monetary"].rank(pct=True) * 0.25
    + cluster_summary["Frequency"].rank(pct=True) * 0.20
    + cluster_summary["CustomerLifetimeDays"].rank(pct=True) * 0.15
    + cluster_summary["TotalOrders"].rank(pct=True) * 0.05
)

cluster_summary = (
    cluster_summary
    .sort_values(
        "BusinessScore",
        ascending=False
    )
)

print("\nCluster Summary")
print("-" * 50)
print(cluster_summary)

# نامگذاری خوشه ها
segment_labels = {}

clusters = cluster_summary.index.tolist()

for i, cluster in enumerate(clusters):
    if i == 0:
        segment_labels[cluster] = (
            "VIP Customers"
        )
    elif i == 1:
        segment_labels[cluster] = (
            "Loyal Customers"
        )
    elif i == len(clusters) - 1:
        segment_labels[cluster] = (
            "Low Value Customers"
        )
    else:
        segment_labels[cluster] = (
            "Regular Customers"
        )

# تشخیص مشتریان جدید
new_customer_cluster = (
    cluster_summary[
        [
            "CustomerLifetimeDays",
            "TotalOrders"
        ]
    ]
    .sum(axis=1)
    .idxmin()
)

segment_labels[
    new_customer_cluster
] = "New Customers"

# برچسب گذاری خوشه ها
df["CustomerSegment"] = (
    df["Cluster"]
      .map(segment_labels)
)

# آمار خوشه ها
print("\nCustomer Segment Counts")
print("-" * 50)

print(
    df["CustomerSegment"]
      .value_counts()
)

# خلاصه خوشه ها
segment_summary = (
    df.groupby("CustomerSegment")
      [
          [
              "TotalRevenue",
              "TotalOrders",
              "Frequency",
              "Monetary"
          ]
      ]
      .mean()
      .round(2)
)

print("\nSegment Summary")
print("-" * 50)
print(segment_summary)

# نمودار خوشه ها
# تعداد اعضای خوشه ها
plt.figure(figsize=(12, 6))
segment_order = (
    df["CustomerSegment"]
      .value_counts()
      .index
)
sns.countplot(
    data=df,
    x="CustomerSegment",
    order=segment_order
)
plt.title(
    f"Customer Segmentation (K={best_k})"
)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# نمودار مقدار خرید و تعداد خرید ها
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x="TotalRevenue",
    y="TotalOrders",
    hue="CustomerSegment",
    s=100
)
plt.title(
    "Customer Segments"
)
plt.tight_layout()
plt.show()

# ذخیره نتایج در دیتابیس جهت داشبورد
result = df[
    [
        "CustomerKey",
        "CustomerName",
        "CustomerSegment",
        "Cluster",
        "TotalRevenue",
        "TotalOrders",
        "AverageOrderValue",
        "Frequency",
        "Monetary",
        "CustomerLifetimeDays"
    ]
]

result.to_sql(
    "CustomerSegmentationML",
    engine,
    if_exists="replace",
    index=False
)

print("\nResults saved successfully.")
print("Table: CustomerSegmentationML")

from sqlalchemy import create_engine
import pandas as pd
# اتصال یه دیتابیس و خواندن داده ها
engine = create_engine(
    "mssql+pyodbc://localhost/AdventureWorksDW2025?"
    "driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
)

df = pd.read_sql(
    """
    SELECT *
    FROM dbo.vw_MonthlyRevenueTrend
    ORDER BY CalendarYear, MonthNumberOfYear
    """,
    engine
)
# فروش 3 ماه قبل از ماه جاری
df["Lag1"] = df["TotalSales"].shift(1)
df["Lag2"] = df["TotalSales"].shift(2)
df["Lag3"] = df["TotalSales"].shift(3)
# میانگین متحرک 3 ماهه
df["RollingMean3"] = (
    df["TotalSales"]
      .rolling(3)
      .mean()
)
# انحراف معیار سه ماهه
df["RollingStd3"] = (
    df["TotalSales"]
      .rolling(3)
      .std()
)
# حذف 3 ردیف اول چون میانگین 3 ماهه و فروش 3 ماه قبل null می باشد
df.dropna(inplace=True)

features = [
    "Lag1",
    "Lag2",
    "Lag3",
    "RollingMean3",
    "RollingStd3"
]
# فیچرها و مقدار پیش بینی برای مدل
X = df[features]
y = df["TotalSales"]
# تقسیم بندی برای داده های آموزش و تست
split = int(len(df) * 0.8)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

from sklearn.linear_model import LinearRegression
# مدل خطی ساده
lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)
lr_pred = lr.predict(
    X_test
)

from sklearn.ensemble import RandomForestRegressor
# 300 درخت تصمیم جهت ساخت مدل
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)
rf_pred = rf.predict(
    X_test
)

from xgboost import XGBRegressor
# بهترین مدل
xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

xgb.fit(
    X_train,
    y_train
)
xgb_pred = xgb.predict(
    X_test
)

from sklearn.metrics import mean_absolute_error
# ارزیابی مدل با محاسبه مقدار خطا
print(
    "Linear Regression:",
    mean_absolute_error(
        y_test,
        lr_pred
    )
)

print(
    "Random Forest:",
    mean_absolute_error(
        y_test,
        rf_pred
    )
)

print(
    "XGBoost:",
    mean_absolute_error(
        y_test,
        xgb_pred
    )
)
# پیش بینی مقدار فروش ماه آینده
last_row = X.iloc[[-1]]
next_month_sales = (
    xgb.predict(last_row)[0]
)

print(
    f"Forecast Next Month Revenue: "
    f"${next_month_sales:,.0f}"
)

import matplotlib.pyplot as plt
import seaborn as sns
# نمودار پیش بینی
plt.figure(figsize=(14,6))
sns.lineplot(
    x=y_test.index,
    y=y_test,
    label="Actual"
)
sns.lineplot(
    x=y_test.index,
    y=lr_pred,
    label="lr_pred"
)
sns.lineplot(
    x=y_test.index,
    y=rf_pred,
    label="rf_pred"
)
sns.lineplot(
    x=y_test.index,
    y=xgb_pred,
    label="xgb_pred"
)


plt.title(
    "Actual vs Forecast Revenue"
)

plt.show()

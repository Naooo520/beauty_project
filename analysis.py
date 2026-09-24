
# %% 讀取資料
# python -m pip install pandas
import pandas as pd
df = pd.read_csv( "杏仁酸資料final.csv")

print(df.head())
print(df.shape)
print(df.columns.tolist())

# %% 切割2/3訓練集與1/3驗證集
# python -m pip install scikit-learn
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df,
    test_size=1/3,
    random_state=42,
    shuffle=True
)

print("訓練集：", train_df.shape)
print("驗證集：", val_df.shape)

# %% 切割輸入矩陣X_n*m、輸出矩陣Y_n*1(售價)
X_train = train_df.drop(columns=["售價(NT)", "每ml價格"])
Y_train = train_df[["售價(NT)"]]

X_val = val_df.drop(columns=["售價(NT)", "每ml價格"])
Y_val = val_df[["售價(NT)"]]

print("X_train：", X_train.shape)
print("Y_train：", Y_train.shape)
print("X_val：", X_val.shape)
print("Y_val：", Y_val.shape)


# %% check缺失值
print("資料型態：")
print(X_train.dtypes)

print("\nX_train 缺失值：")
print(X_train.isna().sum())

print("\nY_train 缺失值：")
print(Y_train.isna().sum())

missing = X_train.isna().sum()
print(missing[missing > 0])

# %% 處理缺失值
X_train["次酸類"] = X_train["次酸類"].fillna("無")
X_train["第三酸類"] = X_train["第三酸類"].fillna("無")

X_val["次酸類"] = X_val["次酸類"].fillna("無")
X_val["第三酸類"] = X_val["第三酸類"].fillna("無")

# %% 確認
print(X_train.isna().sum()[X_train.isna().sum() > 0])
print(X_val.isna().sum()[X_val.isna().sum() > 0])


# %% 區分文字欄位與數字欄位

categorical_cols = X_train.select_dtypes(include=["object", "str"]).columns
numeric_cols = X_train.select_dtypes(include=["number"]).columns

print("文字欄位：")
print(categorical_cols.tolist())

print("\n數字欄位：")
print(numeric_cols.tolist())

# %% 文字欄位轉換成數字

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


# %% 建立決策樹模型
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

model = Pipeline([
    ("preprocessor", preprocessor),
    ("tree", DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    ))
])

# %% 學習
_ = model.fit(X_train, Y_train)

print("模型訓練完成")

# %%預測驗證集售價
Y_pred = model.predict(X_val)

print(Y_pred[:10])

# %% MAE
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(Y_val, Y_pred)

print("MAE：", mae)


# %%# 比較不同決策樹深度的 MAE

from sklearn.metrics import mean_absolute_error


results = []
for depth in range(1, 51):

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("tree", DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        ))
    ])

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_val)

    mae = mean_absolute_error(Y_val, Y_pred)

    results.append((depth, mae))

    print("樹深：", depth, "MAE：", mae)

best_depth, best_mae = min(results, key=lambda x: x[1])

print("\n最佳樹深：", best_depth)
print("最低 MAE：", best_mae) 

# %% 最佳模型
best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("tree", DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=42
    ))
])

best_model.fit(X_train, Y_train)
Y_pred_best = best_model.predict(X_val)
best_mae = mean_absolute_error(Y_val, Y_pred_best)

print("最佳樹深：", best_depth)
print("驗證集 MAE：", best_mae)


#%% NMF 專用資料前處理

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

nmf_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", MinMaxScaler(clip=True), numeric_cols)
    ]
)


# %% NMF先將 X 轉成數值資料

X_train_encoded = nmf_preprocessor.fit_transform(X_train)
X_val_encoded = nmf_preprocessor.transform(X_val)

print("X_train encoded：", X_train_encoded.shape)
print("X_val encoded：", X_val_encoded.shape)

print("X_train encoded 最小值：", X_train_encoded.min())
print("X_val encoded 最小值：", X_val_encoded.min())


# %% 濃縮五特徵
from sklearn.decomposition import NMF

nmf = NMF(
    n_components=5,
    init="nndsvda",
    random_state=42,
    max_iter=1000
)

X_train_nmf = nmf.fit_transform(X_train_encoded)
X_val_nmf = nmf.transform(X_val_encoded)

print("NMF後訓練集：", X_train_nmf.shape)
print("NMF後驗證集：", X_val_nmf.shape)

# %%
from sklearn.linear_model import LinearRegression

nmf_model = LinearRegression()

nmf_model.fit(X_train_nmf, Y_train)

Y_pred_nmf = nmf_model.predict(X_val_nmf)

# %% 算MAE
nmf_mae = mean_absolute_error(Y_val, Y_pred_nmf)

print("NMF + Linear Regression MAE：", nmf_mae)

# %% NMF找最佳MAE
nmf_results = []

for n in range(2, 31):

    nmf = NMF(
        n_components=n,
        init="nndsvda",
        random_state=42,
        max_iter=5000
    )

    X_train_nmf = nmf.fit_transform(X_train_encoded)
    X_val_nmf = nmf.transform(X_val_encoded)

    nmf_model = LinearRegression()
    nmf_model.fit(X_train_nmf, Y_train)

    Y_pred_nmf = nmf_model.predict(X_val_nmf)

    mae = mean_absolute_error(Y_val, Y_pred_nmf)

    nmf_results.append((n, mae))

    print("NMF components：", n, "MAE：", mae)

best_n, best_nmf_mae = min(
    nmf_results,
    key=lambda x: x[1]
)

print("\n最佳 NMF components：", best_n)
print("最低 NMF MAE：", best_nmf_mae)


# %% 
print("Decision Tree 最佳 MAE：", best_mae)
print("NMF + Linear Regression 最佳 MAE：", best_nmf_mae)

if best_mae < best_nmf_mae:
    print("Decision Tree 表現較佳")
else:
    print("NMF + Linear Regression 表現較佳")



# %% 決策樹特徵權重表

tree_model = best_model.named_steps["tree"]

# 取得 One-Hot Encoding 後的特徵名稱
feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

# 取得決策樹特徵重要度
importances = tree_model.feature_importances_

# 建立權重表
importance_df = pd.DataFrame({
    "特徵": feature_names,
    "權重": importances
})

# 轉成百分比
importance_df["權重(%)"] = importance_df["權重"] * 100

# 由大到小排序
importance_df = importance_df.sort_values(
    by="權重",
    ascending=False
).reset_index(drop=True)

# 加入排名
importance_df.index = importance_df.index + 1
importance_df.index.name = "排名"

print(importance_df)

# 取得 One-Hot Encoding 後的欄位名稱
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

# 取得決策樹的特徵重要度
importances = tree_model.feature_importances_

# 整理成 DataFrame
importance_df = pd.DataFrame({
    "特徵": feature_names,
    "重要度": importances
})

# 由大到小排序
importance_df = importance_df.sort_values(
    by="重要度",
    ascending=False
)

print(importance_df)


# %% 將 One-Hot 後的權重加總回原始欄位

# 取得已經訓練完成的 OneHotEncoder
ohe = best_model.named_steps[
    "preprocessor"
].named_transformers_["cat"]

# 建立「轉換後特徵 → 原始欄位」對照
original_feature_names = []

# 類別型欄位
for col, categories in zip(categorical_cols, ohe.categories_):
    original_feature_names.extend(
        [col] * len(categories)
    )

# 數值型欄位
original_feature_names.extend(numeric_cols.tolist())

# 建立表格
original_importance_df = pd.DataFrame({
    "原始欄位": original_feature_names,
    "權重": importances
})

# 同一個原始欄位的 One-Hot 權重加總
original_importance_df = (
    original_importance_df
    .groupby("原始欄位", as_index=False)["權重"]
    .sum()
)

# 轉成百分比
original_importance_df["權重(%)"] = (
    original_importance_df["權重"] * 100
)

# 排序
original_importance_df = (
    original_importance_df
    .sort_values(
        by="權重",
        ascending=False
    )
    .reset_index(drop=True)
)

# 加排名
original_importance_df.index = (
    original_importance_df.index + 1
)
original_importance_df.index.name = "排名"

print("\n原始欄位特徵重要度：")
print(original_importance_df)


# %% 分析不同品牌的預測誤差

error_df = X_val.copy()

error_df["實際售價"] = Y_val.values
error_df["預測售價"] = Y_pred_best

# 每筆資料的絕對誤差
error_df["絕對誤差"] = abs(
    error_df["實際售價"] - error_df["預測售價"]
)

# 依品牌統計
brand_error = (
    error_df
    .groupby("品牌")
    .agg(
        商品數量=("實際售價", "count"),
        平均實際售價=("實際售價", "mean"),
        平均預測售價=("預測售價", "mean"),
        MAE=("絕對誤差", "mean"),
        最大誤差=("絕對誤差", "max")
    )
    .sort_values("MAE", ascending=False)
)

print(brand_error)

# %% 計算各品牌對整體誤差的貢獻

brand_error["總絕對誤差"] = (
    brand_error["MAE"] * brand_error["商品數量"]
)

brand_error["誤差貢獻率(%)"] = (
    brand_error["總絕對誤差"]
    / brand_error["總絕對誤差"].sum()
    * 100
)

brand_error = brand_error.sort_values(
    "總絕對誤差",
    ascending=False
)

print(brand_error)


# %% 分析不同售價區間的預測誤差

error_df["價格區間"] = pd.cut(
    error_df["實際售價"],
    bins=[0, 500, 1000, 1500, 2000, 3000, float("inf")],
    labels=[
        "0~500",
        "501~1000",
        "1001~1500",
        "1501~2000",
        "2001~3000",
        "3000以上"
    ]
)

price_error = (
    error_df
    .groupby("價格區間", observed=True)
    .agg(
        商品數量=("實際售價", "count"),
        平均實際售價=("實際售價", "mean"),
        MAE=("絕對誤差", "mean"),
        最大誤差=("絕對誤差", "max"),
        總絕對誤差=("絕對誤差", "sum")
    )
)

price_error["誤差貢獻率(%)"] = (
    price_error["總絕對誤差"]
    / price_error["總絕對誤差"].sum()
    * 100
)

print(price_error)


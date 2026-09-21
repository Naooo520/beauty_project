# 酸類保養產品價格分析與預測

## 專案介紹

近年保養品市場逐漸出現成分導向的趨勢，消費者在選擇產品時，也越來越重視產品中的實際成分與濃度。
本專案最初以杏仁酸產品作為資料蒐集對象，但在蒐集過程中發現，市面上的酸類保養產品種類很多，例如杏仁酸、甘醇酸、水楊酸、乳酸及複合酸產品等，而且不同酸類之間的使用目的也有部分重疊，因此後續將資料範圍擴展至市面上較常見的酸類保養產品。
同時也發現，即使是相似酸類、濃度或用途的產品，售價仍可能有明顯差異，因此希望透過資料整理與機器學習模型，分析產品特徵與售價之間的關係。

---

## 資料來源

產品資料主要蒐集自：
- 品牌官網
- 代理官網
- 蝦皮商城
- momo
- 屈臣氏
- 康是美
- 寶雅
- Amazon
- PChome

資料以台灣可購買的產品為主。
部分 Amazon 商品以美元計價，使用固定匯率：
`1 USD = 31.90 TWD（2026/08/17）`
統一換算為新台幣。

---

## 資料內容

資料集主要包含以下資訊：
- 品牌
- 品牌國家
- 來源平台
- 產品名稱
- 品名長度
- 產品類型
- 主酸類與濃度
- 次酸類與濃度
- 第三酸類與濃度
- 容量
- 原價
- 售價（NT）
- 每 ml 價格
沒有提供明確酸類濃度的產品不納入最後分析；其他未提供的資訊則不自行推測。

---

## 分析流程

1. 蒐集酸類保養產品資料
2. 使用試算表整理與篩選資料
3. 將整理完成的資料下載為 CSV
4. 使用 pandas 將 CSV 匯入 Python
5. 將資料分為 Training Set 與 Validation Set
6. 檢查並處理缺失值
7. 將類別型資料使用 One-Hot Encoding 轉換
8. 建立 Decision Tree 模型
9. 比較不同 `max_depth` 的 MAE
10. 使用 NMF 進行特徵降維
11. 使用 Linear Regression 建立價格預測模型
12. 使用 MAE 比較模型預測結果

---

## 使用技術

- Python
- pandas
- scikit-learn
- One-Hot Encoding
- Decision Tree Regression
- Non-negative Matrix Factorization（NMF）
- Linear Regression
- Mean Absolute Error（MAE）

---

## 模型結果

本專案比較兩種價格預測方式：

| 模型 | 最佳 MAE |
| --- | ---: |
| Decision Tree | 189.14 |
| NMF + Linear Regression | 300.84 |

Decision Tree 測試不同的 `max_depth` 後，本次資料中較佳的設定為：`max_depth = 27`

最終 Decision Tree 的 MAE 約為 **189.14**，低於 NMF + Linear Regression 的 **300.84**。

MAE 越低代表模型預測價格與實際價格之間的平均誤差越小，因此在本次資料與分析方法下，Decision Tree 的預測誤差較低。

---

## 執行環境

主要使用 Python，所需套件：

```bash
pip install pandas scikit-learn

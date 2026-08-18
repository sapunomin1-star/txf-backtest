# 台指期 EMA／波動風控策略研究

這個 repository 收錄台灣指數期貨（TXF／TMF）的研究、回測、策略監控與 Shioaji 模擬盤工具。版本庫只保留程式與必要的相依套件清單；行情、執行紀錄、策略產出與帳戶狀態不隨 repository 提供。

## 核心策略

目前稽核後的核心邏輯是 **EMA480／EMA2160 趨勢訊號 + RV20 高波動減碼**：

- 以 1 分 K 收盤價計算 EMA480 與 EMA2160。
- EMA480 高於 EMA2160 加上 5 點帶寬時做多，否則空手。
- RV20 使用完整交易日的日收盤報酬，取 20 日標準差並以 `sqrt(252)` 年化。
- RV20 高於 15% 時，目標曝險縮放為原本的 0.5；其餘時間維持完整目標曝險。
- 實際整數口數仍受權益、合約名目價值、槓桿上限與保證金安全檢查約束。

策略參數與市場制度可能隨時間失效。任何結果都必須以樣本外、交易成本、滑價、換月與壓力情境重新驗證。

## 主要入口

| 檔案 | 用途 |
| --- | --- |
| `txf_ema_trend_backtest.py` | 參數化的 TXF 1 分 K EMA 基礎回測。研究 480／2160 組合時請明確傳入 `--fast 480 --slow 2160`；完整 RV20 風控以監控與模擬盤工具為準。 |
| `txf_strategy_monitor.py` | 取得或讀取近期 TAIFEX 資料，計算 EMA480／2160、RV20 與目前目標曝險。這是研究／監控工具，不是實盤下單引擎。 |
| `shioaji_paper_test.py` | 使用 Shioaji 即時資料執行 paper tracking；預設不送單。只有明確指定模擬下單參數時，才會調整 Shioaji 模擬帳戶部位。 |
| `shioaji_txf_kbars_download.py` | 下載 TXFR1 連續近月 1 分 K。 |
| `shioaji_per_contract_download.py` | 探測掛牌 TXF 合約並分合約下載 1 分 K。 |
| `fetch_txo_ivx.py` | 取得選擇權鏈與 IVX 研究資料。 |
| `strategy_adversarial_validation.py` | 進行成本、參數鄰域、跳空與其他對抗式穩健性檢查。 |
| `strategy_optimization.py` | 研究參數與曝險組合；最佳結果仍須另外做樣本外驗證。 |
| `leverage_margin_stress_test.py` | 槓桿、保證金與衝擊情境壓力測試。 |
| `test_strategy_safety.py` | 策略安全條件的測試。 |

根目錄的 shell scripts 提供 Shioaji 環境設定、資料下載與每日監控的執行包裝。

## 安裝

建議使用獨立虛擬環境：

```bash
python3 -m venv .venv-shioaji
source .venv-shioaji/bin/activate
python3 -m pip install -r requirements_shioaji.txt
```

其他研究腳本亦使用 NumPy 等資料科學套件；請依實際入口補齊本機環境中的相依套件。

## Shioaji 憑證安全

Shioaji API 金鑰只應透過執行環境提供：

```bash
export SHIOAJI_API_KEY="your_api_key"
export SHIOAJI_SECRET_KEY="your_secret_key"
```

請勿把真實金鑰寫入程式、README、shell script、`.env`、執行紀錄或 Git history。repository 的 ignore 規則會排除 `.env*`、常見憑證格式、log 與本地資料目錄，但 ignore 規則不能取代提交前的人工檢查。

## 資料與本機路徑

本 repository 不提供 FinMind、TAIFEX、Shioaji 或其他第三方行情資料。使用者需自行取得資料、確認使用與再散布權利，並透過各腳本的命令列參數指定輸入與輸出位置。

部分研究腳本仍保留開發機器上的絕對路徑作為預設值。換到其他環境前，請改為相對路徑、設定檔或命令列參數；不要直接依賴這些本機預設值。

以下目錄預設不納入版本控制：

- `finmind_data/`
- `shioaji_data/`
- `taifex_prev30/`
- `output/`
- `logs/`
- Python 虛擬環境與快取

## 免責聲明

本專案僅供程式研究、教育與策略驗證，不構成投資建議、招攬、保證獲利或適合任何人的交易方案。期貨與槓桿交易可能快速造成重大損失，歷史回測與模擬結果不代表未來績效。使用者須自行檢查資料品質、程式正確性、交易成本、滑價、流動性、保證金、券商與交易所規則，並自行承擔所有決策與損失。

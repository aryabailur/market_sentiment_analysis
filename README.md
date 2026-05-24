# Trading Behavior & Market Sentiment Analysis



## What this is

I analyzed ~211K trades from Hyperliquid and cross-referenced them with the Fear/Greed index to see how market sentiment actually affects how traders behave and perform. The goal was to go beyond surface-level correlations and find patterns that are actually useful.

## Datasets

**Historical Trader Data:** 211,224 rows of on-chain trade records from Hyperliquid. Key columns: account address, asset, execution price, size (tokens + USD), side (buy/sell), timestamp, start position, closed PnL, fee.

**Fear/Greed Index:** one record per day with a score from 0 to 100 and a category label (Extreme Fear to Extreme Greed).

## How I set it up

- Converted timestamps in both datasets to proper datetime format
- Extracted date-only column from the trader data to merge on
- Merged both datasets on date and ended up with 211,218 rows (lost 6 rows because Oct 26 2024 had no Fear/Greed record, which is fine)
- Filtered to closed trades only (Closed PnL != 0) giving 104,402 rows
- Created a pnl_efficiency column = Closed PnL / Size USD to measure return per dollar deployed

## Findings

**Trade volume is highest during Fear, not Greed**

Fear: 61,837 trades. Extreme Fear: 21,400. Extreme Greed: 39,992.

Traders are more active when markets are falling than when they're rising. This suggests people are reacting to volatility rather than trading with confidence.

**Mean PnL looks good. Median tells a different story.**

Median PnL is $0 across every sentiment category. More than half of all trades close at breakeven or a loss, regardless of what the market is doing. The mean is being pulled up by a small number of traders making very large profits. It doesn't reflect the typical trader's experience at all.

**Extreme Greed = smaller bets, much better returns per dollar**

During Extreme Greed, average trade size is $2,779 and median PnL efficiency is 0.027 (2.7 cents per dollar risked).
During Fear, average trade size jumps to $8,041 but efficiency drops to 0.008.

Traders are betting 3x more during Fear and getting worse results. During Extreme Greed they're being more selective and it shows.

## Trader Segmentation

Looked at the 32 accounts with 20+ trades and grouped them:

| Segment                           | Win Rate | Efficiency | Avg Trades |
| --------------------------------- | -------- | ---------- | ---------- |
| Consistent Winner (9 accounts)    | 97.3%    | 0.175      | 2,158      |
| Average Performer (11 accounts)   | 81.5%    | 0.045      | 5,205      |
| Sentiment Dependent (10 accounts) | 82.2%    | 0.028      | 1,940      |
| Consistent Loser (2 accounts)     | 63.2%    | 0.008      | 4,163      |

Consistent Losers average more trades than almost every other segment. More trading isn't the answer.

The most interesting group is Sentiment Dependent. Some of these are massive earners. 0xbaaa... has a 99.1% win rate across nearly 10,000 trades but drops to 33% during Greed. 0x0833... is the #2 account by total PnL ($1.6M) but collapses to 26% win rate during Extreme Greed. Their edge is real, but it only works in specific conditions.

## Model

Built an XGBoost classifier to predict whether a trade would be profitable.

Features: sentiment category, raw fear/greed score, trade direction, position size, account historical win rate.

One thing I caught during this: I initially included pnl_efficiency as a feature and got ROC-AUC of 1.0. That's data leakage since efficiency is calculated directly from Closed PnL (the target). I dropped it and retrained.

**Final result: ROC-AUC 0.89, Accuracy 80%**

Feature importance:

- account_win_rate = 0.363
- value (raw score) = 0.226
- side_encoded = 0.187
- sentiment_encoded = 0.161
- Size USD = 0.062

The raw sentiment score (0 to 100) was more predictive than the category label, which makes sense. Whether the index is at 78 vs 92 probably matters more than both being labeled "Extreme Greed."

Trader identity is the strongest single predictor, but sentiment features combined (0.387) actually edge it out. So sentiment isn't just noise. It has real independent influence on whether a trade wins.

Install dependencies with `pip install -r requirements.txt`

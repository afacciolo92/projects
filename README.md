# Projects

A collection of data science, machine learning, and research projects by Andrew Facciolo.

---

## Airbnb Analysis

A 12-month study of New York City's short-term rental market using Inside Airbnb data from June 2024 through May 2025. The project merges three datasets — listings, calendar availability, and neighbourhoods — through a reproducible batch-processing pipeline that cleans, standardizes, and consolidates each month into a single Parquet file for analysis.

**What the analysis covers:**
- Seasonal price trends across NYC, revealing summer/fall peaks and post-holiday dips
- Neighbourhood-level price comparison across the top 5 neighbourhoods by listing volume (e.g. Midtown luxury pricing vs. Bedford-Stuyvesant budget rates)
- Monthly occupancy rate estimation derived from calendar availability flags
- Price vs. occupancy scatter analysis across neighbourhoods and multiple time periods, surfacing potential market oversupply and pricing misalignment

**Key findings:** Average nightly prices peak in October (~$494) and bottom out in January. High-price neighbourhoods like Midtown consistently show low occupancy, while budget-friendly areas maintain stronger demand — suggesting localized oversaturation in the luxury segment.

**Tools:** Python, pandas, NumPy, Matplotlib, Seaborn, PyArrow

[Report](airbnb-analysis/Airbnb%20Analysis.pdf) | [Code](airbnb-analysis/Airbnb_Analysis_Code.py)

---

## Credit Default Compass

An end-to-end machine learning pipeline for predicting credit card defaults using the UCI Credit Card dataset (30,000 clients, 25 features). The project walks through the full modelling lifecycle — from EDA and data cleaning through feature engineering, model training, hyperparameter tuning, and threshold optimization for deployment.

**What the pipeline includes:**
- Data cleaning: recoding invalid category labels (EDUCATION, MARRIAGE), winsorizing extreme outliers at the 1st/99th percentiles, and capping repayment delay codes
- Feature engineering: credit utilization ratios, payment coverage ratios, aggregate delay summaries (max, mean, ever-late flag), delay severity buckets, and month-over-month bill/payment trends
- Model comparison: Logistic Regression vs. Random Forest, both with balanced class weights and GridSearchCV hyperparameter tuning (5-fold cross-validation)
- Threshold analysis: F1-optimal, recall-targeted (60%/65%/70%), and cost-optimal (10:1 missed-default-to-false-alarm ratio) cutoffs evaluated with precision-recall curves and confusion matrices

**Key results:** The tuned Random Forest achieved a test-set ROC AUC of 0.774, outperforming Logistic Regression (0.755). At the 60%-recall threshold, the model catches 6 in 10 defaulters with 47.5% precision and an 18.8% false-positive rate. Feature importance analysis confirms that recent repayment delay (PAY_0) and maximum delay history are the strongest predictors.

**Tools:** Python, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

[Report](credit-default-compass/Credit%20Default%20Compass.pdf) | [Code](credit-default-compass/Credit_Default_Code.py)

---

## Olist Seller Ranking Prioritization Experiment

A simulated A/B test on the Olist Brazilian e-commerce dataset designed to answer: can a data-driven seller ranking improve customer experience? The project merges nine relational datasets (orders, items, customers, sellers, products, payments, reviews, geolocation, categories) into an order-level analytical frame, then builds and evaluates a seller prioritization system.

**What the experiment covers:**
- Data integration: joining nine CSVs into a unified order-level summary with derived fields (customer tenure, delivery speed, order value, review scores)
- Seller scoring: z-scored average review rating and delivery speed combined into a composite priority score, then ranked per product (top-K = 3)
- Experiment design: stratified 50/50 customer split on product category, validated with balance checks (means, distributions), Welch's t-tests, and Cohen's d to confirm no pre-existing bias
- Simulation: control group sees the original seller; test group sees the highest-priority seller per product, with group-level outcome comparison on review scores and delivery times

**Tools:** Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn

[Report](olist-seller-ranking/Olist%20Seller%20Ranking%20Prioritization%20Experiment.pdf) | [Code](olist-seller-ranking/Olist_code_final.py)

---

## Solar PV as a Non-Wires Alternative in Ontario

Research paper examining the viability of solar photovoltaic systems as a non-wires alternative within Ontario's electricity planning framework. Evaluates how distributed solar generation could defer or replace traditional infrastructure investments in transmission and distribution.

[Paper](solar-pv-ontario/)

---

## Fact-Checking of Statements from Political Figures

An analytical report evaluating the accuracy of claims made by political figures, cross-referencing statements against available data and public sources.

[Report](fact-checking-political-figures/)

---

## Trading Bot — LLM-Driven Sentiment Trading Agent

An LLM-powered trading bot (LeafTrade) that combines fundamental screening, technical analysis, and GPT-4o-mini sentiment scoring to trade S&P 500 stocks via the Alpaca paper trading API. Designed for student investors seeking inflation-beating returns with disciplined risk management.

**What the pipeline includes:**
- 5-criteria funnel: quality screen (10yr return, Sharpe), liquidity filter, risk/volatility screen, technical trend confirmation (MA50, RSI), and LLM sentiment gating (score, consensus, headline count)
- Risk-based position sizing: 2% portfolio risk per trade, 15% position cap, ATR-based trailing stops
- Full backtesting suite with portfolio simulation, parameter optimization, and equity curve analysis
- Live paper trading mode with Alpaca integration, bracket orders, and demo mode for presentations

**Key results:** Backtested strategy achieved a 1.31 Sharpe ratio and 19.2% annualized return over the test period, outperforming a buy-and-hold S&P 500 baseline on a risk-adjusted basis.

**Tools:** Python, OpenAI GPT-4o-mini, Alpaca API, pandas, NumPy, Matplotlib, yfinance, NewsAPI

[Report](trading-bot/LLM-Driven-Trading-Agent-LeafTrade.pdf) | [Code](trading-bot/) | [README](trading-bot/README.md)

---

## Toronto KSI Collision Hotspot Dashboard

An interactive Tableau dashboard analyzing Killed or Seriously Injured (KSI) collision events across Toronto (2006–2023), built from City of Toronto Open Data Portal records. Designed for city executives and public safety planners who need actionable, location-specific intelligence to allocate enforcement resources and evaluate road safety policy.

**What the dashboard covers:**
- Geographic hotspot map of the 20 most dangerous intersections by cumulative KSI count, with proportional symbol encoding
- Six cross-filters (year range, time of day, four contributing-factor toggles) driving simultaneous updates across all views
- Time-of-day decomposition across five periods (Night, AM Peak, PM Peak, Midday, Overnight)
- Executive KPI tiles showing total KSI count and contributing-factor shares, updated dynamically with every filter change

**Key findings:** Aggressive driving is flagged in 50.7% of KSI collisions at the Top 20 hotspot intersections. Night hours (6 PM–midnight) carry the largest risk share at 29.85%. Finch Ave W & Weston Rd consistently ranks as the single most dangerous hotspot.

**Tools:** Tableau Desktop, City of Toronto Open Data Portal

[Report](toronto-ksi-dashboard/Toronto_KSI_Collision_Hotspot_Report.pdf) | [Dashboard](toronto-ksi-dashboard/) | [README](toronto-ksi-dashboard/README.md)

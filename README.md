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

---

## When the Platform Shapes the Person: An Ethical Analysis of AI-Enhanced Surveillance Platforms

A graduate paper examining AI-enhanced surveillance platforms — social media systems, search engines, and adjacent digital environments whose value lies less in the service they appear to provide than in the behavioural data they capture and the predictions they generate. The argument moves beyond familiar concerns about misinformation and privacy to ask a structural question about the kind of human and civic world this technology presupposes and produces.

**What the paper covers:**
- The shift from communication tool to behavioural infrastructure, where social interaction is folded into a commercial logic that converts human experience into a "predictive commodity" (Lanchester; Zuboff & Klein)
- James Scott's *Seeing Like a State* applied to algorithmic legibility — how users are simplified into traces, patterns, and ranked probabilities in order to be governable and monetizable
- Martin Buber's *I and Thou* applied to platform design, framing surveillance platforms as an institutionalized I–It mode of relation where the user is approached as segment, target, or attention reservoir rather than as a person
- Don Tapscott's framing of digital infrastructure as a public-trust question, used to argue that surveillance-based business models deform civic deliberation, not just individual privacy
- A normative argument that the deepest harm is structural: incentives that reward reaction over reflection, immediacy over judgment, and engagement over truth

**Tools:** Long-form ethical analysis grounded in Zuboff & Klein, Lanchester, Scott, Buber, and Tapscott

[Paper](ai-surveillance-ethics/Ethical%20Analysis%20of%20AI-Enhanced%20Surveillance%20Platforms.pdf)

---

## DETR Object Detection — Candy Counter

A fine-tuning project on Facebook's DETR (DEtection TRansformer) with a ResNet-50 backbone, adapted for a custom 8-class candy detection task. The notebook walks through the full pipeline — from converting Label Studio's COCO export into a Hugging Face `datasets`-compatible format, through transfer learning, evaluation, and packaging the model behind a deployable inference function.

**What the pipeline includes:**
- COCO → Hugging Face conversion: parsing `result.json`, attaching per-image bounding boxes and category IDs, building `id2label` / `label2id` maps, writing JSONL metadata, and producing an `imagefolder` dataset with an 80/20 train/validation split
- Fine-tuning: `facebook/detr-resnet-50` checkpoint loaded via `DetrForObjectDetection` with `ignore_mismatched_sizes=True` to swap in the new classification head; the image processor handles bbox encoding via `with_transform`
- Training loop: a version-agnostic `TrainingArguments` builder that probes the installed Transformers version for supported keys (`evaluation_strategy` vs `eval_strategy`, FP16 availability, etc.), then trains via Hugging Face `Trainer` with per-epoch evaluation and best-model loading
- Inference: a `candy_counter()` function that accepts a NumPy or PIL image, runs the saved model, post-processes detections at a tunable threshold, and returns per-class counts across 8 candy types (Moon, Insect, Black_star, Grey_star, Unicorn_whole, Unicorn_head, Owl, Cat)

**Tools:** Python, PyTorch, Hugging Face Transformers + Datasets, DETR (ResNet-50), PIL, Google Colab

[Notebook](detr-object-detection/DETR%20Object%20Detection%20-%20Candy%20Counter.ipynb)

---

## Neural Network from Scratch — NumPy Backpropagation

A 3-layer fully connected neural network implemented end-to-end in pure NumPy — no TensorFlow, PyTorch, or autodiff. Every layer, activation, loss, and optimizer is written by hand, with explicit forward and backward passes that make the gradient math visible.

**What's implemented from scratch:**
- `DenseLayer`: weight init (`N(0,1) * 0.01`), bias init at zero, forward computes `z = X @ W + b` and caches inputs, backward computes `dW`, `db`, and `dinputs` via the chain rule
- `ReLu`: forward `max(0, z)`, backward zeros out the gradient where `z ≤ 0`
- `Softmax`: row-wise softmax with the standard max-subtraction trick for numerical stability; backward implements the full Jacobian-vector product (`diag(p) − p pᵀ`) per row
- `CrossEntropyLoss`: clipped one-hot cross-entropy with batch-size-normalized gradients
- `SGD`: per-layer parameter update (`W ← W − η · dW`, `b ← b − η · db`)
- Mini-batch training loop with per-epoch shuffling, one-hot encoding, and a 3–4–8–3 architecture trained on a 3-class dataset with a held-out test split

**Tools:** Python, NumPy

[Code](neural-network-from-scratch/Neural%20Network%20from%20Scratch%20-%20NumPy%20Backpropagation.py)

---

## LSTM and GRU Time-Series Forecasting

An RNN-based demand forecasting pipeline that compares LSTM and GRU architectures against naive and moving-average baselines, evaluated on both statistical accuracy (RMSE) and an asymmetric-cost business metric.

**What the pipeline includes:**
- Data prep: time-based 80/20 train/val split, MinMax scaling fit on training only, and a 7-day lookback window converting the 1D demand series into supervised sequences for the RNN
- Model comparison: simple `LSTM(32) → Dense(1)` and `GRU(32) → Dense(1)` architectures with identical training (Adam, MSE loss, 20 epochs, batch size 16); the lower-RMSE model is selected automatically
- Baselines: naive `D̂[t+1] = s[t]` and a 7-day moving average, aligned to the same validation window for apples-to-apples comparison
- Business-metric evaluation: a Total Inventory Cost calculation with asymmetric holding (`$1`) and stockout (`$5`) penalties — a forecast that minimizes RMSE may not minimize cost when over- and under-forecasting are penalized differently

**Tools:** Python, TensorFlow/Keras, scikit-learn, pandas, NumPy

[Code](lstm-gru-forecasting/LSTM%20and%20GRU%20Time-Series%20Forecasting.py)

---

## Customer360 Data Product

A full Customer360 implementation that turns a transactional star schema (orders, conversions, customers, products, dates) into a per-customer, per-week analytical table covering every week between conversions — including weeks with no orders. The output is a single denormalized `customer_info` table that supports cohort, lifetime-value, and conversion-channel analysis directly, without further joins.

**What the implementation covers:**
- Schema design: `customer360.customer_info` with conversion sequencing (`conversion_number`, `conversion_type`, channel), week tracking (`order_week`, `week_counter`, `next_conversion_week`), and cumulative revenue at both per-conversion and lifetime grain
- Multi-CTE pipeline: `order_with_product` enriches the orders fact with product, date, and customer dimensions and derives `price_before_discount`; `customer_and_conversion` joins conversions to customers and uses `RANK() OVER (PARTITION BY customer_id ORDER BY conversion_id)` for conversion ordering and `LEAD(...)` to look ahead to the next conversion week
- Weekly sequence generation: `week_count` enumerates every distinct week in the date dimension; `conv_with_counter` cross-joins each conversion to all subsequent weeks and uses `ROW_NUMBER() OVER (PARTITION BY customer_id, conversion_number)` to attach a per-conversion week counter; `conv_cust_weekCount` filters to weeks before the next conversion (or end-of-data when none exists)
- Final insert: weekly orders aggregated in `grouped_orders`, joined back, and `SUM(...) OVER (PARTITION BY ... ORDER BY week_counter)` window functions compute conversion-level and lifetime-level cumulative revenue running totals

**Tools:** SQL (T-SQL) — multi-CTE design, RANK, ROW_NUMBER, LEAD, window-function running sums

[Report](customer360/Customer360%20Implementation%20Report.pdf) | [SQL](customer360/Customer360_Implementation.sql)

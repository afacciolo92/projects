# %%
from pathlib import Path
import pandas as pd
import os

# %%
# point at the data directory, one level up from 'notebooks/'
DATA_DIR = Path().resolve().parent.parent / "data"

# %%
# 1) inspect where we are running:
print("cwd =", os.getcwd())

# 2) point to the data folder one level up
DATA_DIR = Path(os.getcwd()) / "data"


# %%
# 1) Where are we?
print("Working directory:", os.getcwd())

# 2) Define your data folder: one level up from notebooks/, then data/archive/
DATA_DIR = Path.cwd().parent / "data" / "archive"
assert DATA_DIR.exists(), f"Could not find data folder at {DATA_DIR}"

print("Loading CSVs from:", DATA_DIR)

# %%
# ┌── Cell 2: Load all Olist CSVs ───────────────────────────────────────────────
customers      = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
geolocation    = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
order_items    = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
order_payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
order_reviews  = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
orders         = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
products       = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
sellers        = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")
categories     = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")

# %%
# Quick sanity: print shapes
for df, name in [
    (customers, "customers"),
    (geolocation, "geolocation"),
    (order_items, "order_items"),
    (order_payments, "order_payments"),
    (order_reviews, "order_reviews"),
    (orders, "orders"),
    (products, "products"),
    (sellers, "sellers"),
    (categories, "categories"),
]:
    print(f"{name:15s} → {df.shape}")

# %%
# ┌── Cell 3: Peek at the first few rows of each ────────────────────────────────
display(customers.head(2))
display(geolocation.head(2))
display(order_items.head(2))
display(order_payments.head(2))
display(order_reviews.head(2))
display(orders.head(2))
display(products.head(2))
display(sellers.head(2))
display(categories.head(2))

# %%
#    we'll join orders → order_items → customers

# 1) orders + items
orders_items = orders.merge(order_items, how="left", on="order_id")
print("orders_items:", orders_items.shape)


# %%
# 2) attach customer info
orders_items_cust = orders_items.merge(customers, how="left", on="customer_id")
print("orders_items_cust:", orders_items_cust.shape)

# %%
# 3) attach product info (if needed for cohorting by product category)
orders_full = orders_items_cust.merge(
    products[["product_id","product_category_name"]],
    how="left",
    on="product_id"
)
print("orders_full:", orders_full.shape)

# Now orders_full has one row per item‐sold, with order, customer and product category.
# Can pivot it to order‐level, define your test/control split, etc.

# %%
# 4) attach seller info
orders_full = orders_full.merge(
    sellers[[
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state"
        # add any other seller attributes I need here (potentially use later)
    ]],
    how="left",
    on="seller_id"
)
print("orders_full w/ sellers:", orders_full.shape)

# Now orders_full has one row per item-sold, with order, customer, product-category AND seller info.

# %%
# 1) PARSE DATES & COMPUTE TENURE + DELIVERY SPEED
orders_full['order_purchase_timestamp'] = pd.to_datetime(orders_full['order_purchase_timestamp'])
orders_full['order_delivered_customer_date'] = pd.to_datetime(orders_full['order_delivered_customer_date'])

# first‐ever order per customer
cust_first = (
    orders_full
      .groupby('customer_unique_id')['order_purchase_timestamp']
      .min()
      .rename('first_order_ts')
)
orders_full = orders_full.merge(cust_first, on='customer_unique_id', how='left')

# tenure in days
orders_full['customer_tenure_days'] = (
    orders_full['order_purchase_timestamp']
    - orders_full['first_order_ts']
).dt.days

# days between purchase and delivery
orders_full['order_deliver_days'] = (
    orders_full['order_delivered_customer_date']
    - orders_full['order_purchase_timestamp']
).dt.days


# 2) MERGE PAYMENT INFO
payments = pd.read_csv(DATA_DIR/'olist_order_payments_dataset.csv')
payment_agg = (
    payments
      .groupby('order_id')['payment_value']
      .sum()
      .rename('order_value')
)
orders_full = orders_full.merge(payment_agg, on='order_id', how='left')


# 3) MERGE REVIEW SCORES
reviews = pd.read_csv(DATA_DIR/'olist_order_reviews_dataset.csv')
orders_full = orders_full.merge(
    reviews[['order_id','review_score']],
    on='order_id',
    how='left'
)

# %%
import numpy as np

# 1) (Re)confirm my columns
print("orders_full columns:", orders_full.columns.tolist())

# 2) Define a safe-mode helper for categorical columns
def safe_mode(s):
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan

# 3) Build order_summary in one pass
order_summary = (
    orders_full
      .groupby(['order_id','customer_unique_id'], as_index=False)
      .agg(
          n_items              = ('order_item_id',         'size'),
          order_value          = ('order_value',           'first'),
          avg_price_per_item   = ('price',                 'mean'),
          seller_count         = ('seller_id',             'nunique'),
          product_category     = ('product_category_name', safe_mode),
          seller_state         = ('seller_state',          safe_mode),
          customer_state       = ('customer_state',        safe_mode),
          customer_tenure_days = ('customer_tenure_days',  'max'),
          order_deliver_days   = ('order_deliver_days',    'mean'),
          review_score         = ('review_score',          'mean'),
      )
)

print("order_summary shape:", order_summary.shape)
order_summary.head()




# %%
import numpy as np
from sklearn.model_selection import train_test_split

# 1) Build a customer‐level frame for splitting
cust_df = (
    order_summary
      .drop_duplicates('customer_unique_id')
      .loc[:, ['customer_unique_id', 'product_category', 'customer_tenure_days', 'order_value']]
)

# 2) Fill missing product_category so stratify won’t choke on NaN
cust_df['product_category'] = cust_df['product_category'].fillna('MISSING')

# 3) Do the stratified split 50/50
train_ids, control_ids = train_test_split(
    cust_df['customer_unique_id'],
    test_size=0.5,
    random_state=42,
    stratify=cust_df['product_category']
)

# 4) Map groups back onto your order_summary
order_summary['group'] = np.where(
    order_summary['customer_unique_id'].isin(train_ids),
    'test',
    'control'
)

# 5) Quick sanity check
print(order_summary['group'].value_counts())



# %%
# 6) Balance checks
balance = (
    order_summary
      .groupby('group')
      .agg(
         avg_order_value   = ('order_value',          'mean'),
         avg_tenure_days   = ('customer_tenure_days', 'mean'),
         avg_items_per_ord = ('n_items',              'mean'),
         pct_large_orders  = ('order_value', 
                              lambda x: (x > 100).mean())
      )
)
print(balance)


# %%
# 7) Category distribution by group
cat_dist = (
    order_summary
      .groupby(['group','product_category'])
      .size()
      .unstack(fill_value=0)
)
print(cat_dist)


# %%
# Quick recap
# Group sizes: Test: 49,743, Control: 49,698 - Basically a 50/50 split within 0.05%
# Balance on continous metrics: The means are very similar. No major drift.
# Balance on categorical metrics: The distribution of product categories is similar across groups. Tells us that our stratification on product category worked as intended.


# %%
# Let's now visualize with Bar Chart of test VS control counts (hould be 50/50 split)
# With boxplots of order_value by group
# Histogram of customer_tenure_days by group
# Stacked bar showing the share of eah product category by group
# Supporting my balance tables above

import matplotlib.pyplot as plt
import seaborn as sns

# 1) Group sizes
order_summary['group'].value_counts().plot.bar(title="Test vs Control Counts")
plt.ylabel("Number of Customers")
plt.show()

# 2) Order value boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x='group', y='order_value', data=order_summary)
plt.title("Order Value by Group")
plt.ylim(0, order_summary['order_value'].quantile(0.95))  # cap outliers
plt.show()

# 3) Customer tenure distributions
plt.figure(figsize=(6,4))
sns.kdeplot(data=order_summary, x='customer_tenure_days', hue='group', common_norm=False)
plt.title("Customer Tenure (days) by Group")
plt.show()

# 4) Product category mix
cat_dist = (order_summary
            .groupby(['group','product_category'])
            .size()
            .unstack(fill_value=0))
cat_dist.div(cat_dist.sum(axis=1), axis=0).plot.bar(stacked=True, figsize=(12,4))
plt.title("Product Category Share by Group")
plt.ylabel("Proportion")
plt.legend( [], [])
plt.show()


# %%
# Print the above product category share group legend with above depiction just to make the categories clear
cat_dist.div(cat_dist.sum(axis=1), axis=0).plot.bar(stacked=True, figsize=(12,4))
plt.title("Product Category Share by Group")

# %%
from scipy.stats import ttest_ind

# 1) Compute group means
means = order_summary.groupby('group')[['review_score','order_deliver_days']].mean().rename(columns={
    'review_score': 'mean_review_score',
    'order_deliver_days': 'mean_deliver_days'
})
print("Group means:\n", means, "\n")


# %%
# 2) extract the two series
test_rev   = order_summary.loc[order_summary['group']=='test',   'review_score'].dropna()
control_rev= order_summary.loc[order_summary['group']=='control','review_score'].dropna()

test_del   = order_summary.loc[order_summary['group']=='test',   'order_deliver_days'].dropna()
control_del= order_summary.loc[order_summary['group']=='control','order_deliver_days'].dropna()


# %%
# 3) run Welch’s t-tests
t1, p1 = ttest_ind(test_rev,   control_rev,   equal_var=False)
t2, p2 = ttest_ind(test_del,   control_del,   equal_var=False)

print(f"Review Score — t = {t1:.3f}, p = {p1:.3f}")
print(f"Deliver Days  — t = {t2:.3f}, p = {p2:.3f}")


# %%
# Both p-values are well above 0.05 
# so there’s no statistically significant difference in either mean review scores or mean delivery times between test and control groups before running the experiment. 
# That confirms randomization (stratified 50/50 on product category) has produced balanced cohorts—no pre-existing bias.

# %% [markdown]
# 

# %%
# 1) Extract the two samples again
test_rev = order_summary.loc[order_summary['group']=='test',    'review_score']
ctrl_rev = order_summary.loc[order_summary['group']=='control', 'review_score']

test_del = order_summary.loc[order_summary['group']=='test',    'order_deliver_days']
ctrl_del = order_summary.loc[order_summary['group']=='control', 'order_deliver_days']

# 2) Define your Cohen's d function
def cohens_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)

# 3) Compute effect sizes
d_rev = cohens_d(test_rev, ctrl_rev)
# for delivery days we invert the sign so "positive d" always means "better"
d_del = cohens_d(ctrl_del, test_del)

print(f"Cohen's d (Review Score): {d_rev:.3f}")
print(f"Cohen's d (Delivery Days): {d_del:.3f}")


# %%
# Those effect‐sizes are essentially zero—|d|≈0.01—which confirms that at baseline my test and control groups are practically identical on both review score and delivery speed. 
# In other words, there’s no meaningful pre-existing bias between them.

 

# %%
# Seller baseline performance
# group orders_full by seller
seller_perf = (
    orders_full
      .groupby('seller_id', as_index=True)
      .agg(
         avg_rev = ('review_score',       'mean'),
         avg_del = ('order_deliver_days', 'mean')
      )
)

# inspect
print("seller_perf (head):")
display(seller_perf.head())



# %%
# will turn into a single priority score that we can use to rank
# 1. Z-scoring each metric so they're on the same scale
# 2. Flipping the delivery-time score (since shorter is better)
# 3. Averaging the two z-scores into one priority score
from scipy.stats import zscore

# STEP 2: Standardize and combine into priority_score
# ───────────

# 2a) Fill any NaNs in avg_del (e.g. sellers with no recorded delivery days)
seller_perf['avg_del'] = seller_perf['avg_del'].fillna(seller_perf['avg_del'].mean())

# 2b) Compute z‐scores
seller_perf['z_rev'] = zscore(seller_perf['avg_rev'])
# we flip avg_del so that faster delivery → higher z‐score
seller_perf['z_del'] = -zscore(seller_perf['avg_del'])

# 2c) Combine equally into a single priority metric
seller_perf['priority_score'] = seller_perf[['z_rev', 'z_del']].mean(axis=1)

# 2d) Take a peek at the top/bottom performers
display(seller_perf.sort_values('priority_score', ascending=False).head(5))
display(seller_perf.sort_values('priority_score').head(5))

# %%
# Lets wire the new priority_score based on review score and delivery speed into our orders_full DataFrame
# We can then rank sellers for each product so you know "who" would be shown first.
# seller_perf is indexed by seller_id and has priority_score
orders_full = orders_full.merge(
    seller_perf['priority_score'].rename('seller_priority'),
    left_on='seller_id',
    right_index=True,
    how='left'
)

print("orders_full w/ priority:", orders_full.shape)
orders_full.head()


# %%
# Rank sellers per product, keeping NaNs as <NA> via the pandas nullable Int64 dtype
rank_series = (
    orders_full
      .groupby('product_id')['seller_priority']
      .rank(ascending=False, method='first')
)
orders_full['seller_rank'] = rank_series.astype('Int64')

# Quick sanity:
print("Total missing ranks:", orders_full['seller_rank'].isna().sum())
print(
    orders_full
      .loc[:, ['product_id','seller_id','seller_priority','seller_rank']]
      .sort_values(['product_id','seller_rank'])
      .head(10)
)



# %%
# We have some missing ranks. Convert any <NA> seller_ranks into “worst” rank for that product:
#    – compute max rank per product
max_ranks = orders_full.groupby('product_id')['seller_rank']\
                       .max()\
                       .rename('max_rank')

#    – join back
orders_full = orders_full.merge(max_ranks, on='product_id')

#    – fill missing as (max_rank + 1), then cast to int
orders_full['seller_rank'] = (
    orders_full['seller_rank']
      .fillna(orders_full['max_rank'] + 1)
      .astype(int)
)

#    – drop the helper column
orders_full.drop(columns='max_rank', inplace=True)

# Quick sanity check:
print("Final missing ranks:", orders_full['seller_rank'].isna().sum())
display(
    orders_full
      .loc[:, ['product_id','seller_id','seller_priority','seller_rank']]
      .sort_values(['product_id','seller_rank'])
      .head(10)
)


# %%
# Decide on how many sellers I'd like to sjow on the product page. Let's do K=3
# Build a lookup of the top-K sellers for each product
K = 3
topk = (
    orders_full
      .loc[:, ['product_id','seller_id','seller_rank']]
      .drop_duplicates()
      .query('seller_rank <= @K')
      .sort_values(['product_id','seller_rank'])
      .groupby('product_id')['seller_id']
      .apply(list)
      .rename('topk_sellers')
)

# Merge that list back onto your order_summary (or orders_full if you prefer)
orders_full = orders_full.merge(topk, on='product_id', how='left')


# %%
# Split customers into test/control as I did before
np.random.seed(42)
# e.g. 50/50 split by customer_unique_id
cust2group = (
    orders_full[['customer_unique_id']]
      .drop_duplicates()
      .assign(
        group=lambda df: np.where(np.random.rand(len(df)) < 0.5,'test','control')
      )
)

orders_full = orders_full.merge(cust2group, on='customer_unique_id')
print(orders_full['group'].value_counts())

# %%
# Serve recommendations in simulation
# Control sees whatever the current default is (assume they see the actual seller they ordered).
# Test sees my new top-K list; we'd simulate that they pick the first element of topk_sellers
def simulate_chosen_seller(row):
    return row['seller_id'] if row['group']=='control' else row['topk_sellers'][0]

orders_full['simulated_seller'] = orders_full.apply(simulate_chosen_seller, axis=1)


# %%
# re-compute your two key metrics—review_score and order_deliver_days—by group, but using the simulated seller:
# Filter to only the rows where the simulated_seller matches the actual seller
# (i.e. only count the orders your algorithm would have allowed to go through)
df_eval = orders_full.query('simulated_seller == seller_id')

# Compute group‐level means
summary = (
    df_eval
      .groupby('group')[['review_score','order_deliver_days']]
      .mean()
      .rename(columns={
        'review_score':'mean_review_score',
        'order_deliver_days':'mean_deliver_days'
      })
)
print(summary)


# %%
outcome_stats = (
    df_eval
      .groupby('group')[['review_score','order_deliver_days']]
      .agg(['mean','std'])
)
print(outcome_stats)

# %%
orders_full['product_id'].unique()[:10]  # see some valid IDs
SAMPLE_PRODUCT_ID = '001e6ad469a905060d959994f1b41e4f'



# %%
# automatically choose the product with the most orders
SAMPLE_PRODUCT_ID = (
    orders_full['product_id']
    .value_counts()
    .index[0]
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# pick a valid product
SAMPLE_PRODUCT_ID = (
    orders_full['product_id']
    .value_counts()
    .index[0]
)

# compute the top 5 sellers
top5 = (
    orders_full[orders_full['product_id'] == SAMPLE_PRODUCT_ID]
      .drop_duplicates('seller_id')
      .nlargest(5, 'seller_priority')  # easier than sort+head
      .set_index('seller_id')['seller_priority']
)

if top5.empty:
    print(f"No sellers found for product {SAMPLE_PRODUCT_ID}")
else:
    plt.figure(figsize=(8,4))
    top5.plot.barh()
    plt.title(f"Top 5 Sellers for Product {SAMPLE_PRODUCT_ID}")
    plt.xlabel("priority_score")
    plt.ylabel("seller_id")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# %%
import pandas as pd

def print_top_sellers_summary(product_id, n=5):
    """
    For a given product_id, show the Top-n sellers sorted by priority_score,
    along with their average review, average delivery days, and computed score.
    """
    # 1) find all unique sellers for this product
    sellers = orders_full.loc[
        orders_full['product_id'] == product_id, 
        'seller_id'
    ].unique()
    
    # 2) pull their metrics from seller_perf
    df = (
        seller_perf
        .loc[sellers, ['priority_score', 'avg_rev', 'avg_del']]
        .rename(columns={
            'avg_rev': 'mean_review_score',
            'avg_del': 'mean_delivery_days'
        })
        .sort_values('priority_score', ascending=False)
        .head(n)
        .reset_index()  # brings seller_id back as a column
    )
    
    # 3) add a rank column
    df.insert(0, 'rank', range(1, len(df) + 1))
    
    # 4) format & display
    display(
        df.style.format({
            'priority_score':      "{:.3f}",
            'mean_review_score':   "{:.2f}",
            'mean_delivery_days':  "{:.1f}"
        }).set_caption(
            f"Top {n} Sellers for Product {product_id}"
        )
    )

# — example usage —
print_top_sellers_summary("aca2eb7d00ea1a7b8ebd4e68314663af", n=5)

# %%
# I want to make sure the table above is more powerful in what it tells you. We need a product with more sellers
# 1) See how many unique sellers each product has
seller_counts = (
    orders_full
      .groupby("product_id")["seller_id"]
      .nunique()
      .sort_values(ascending=False)
)

# 2) Pick the first product with at least 5 sellers
product_with_5_plus = seller_counts[seller_counts >= 5].index[0]
print(f"Using product_id={product_with_5_plus} which has {seller_counts[product_with_5_plus]} sellers")

# 3) Now re-print the Top-5 table for that product
print_top_sellers_summary(product_with_5_plus, n=5)


# %%




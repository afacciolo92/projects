# %%
################################################################# PHASE 1 - SETUP #################################################################

# %%
# 1. Notebook and imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, precision_recall_curve)

# %%
df = pd.read_csv('/Users/andrewfacciolo/Desktop/Homework/UCI_Credit_Card.csv')

# %%
orig_df = pd.read_csv('/Users/andrewfacciolo/Desktop/Homework/UCI_Credit_Card.csv')

# %%
###################################################### PHASE 2 - EDA & CLEANING ######################################################

# %%
# Inspect structure
df.shape, df.info(), df.describe()

# %%
# Some raw takeaways:
# - 30,000 rows, 25 columns so plenty of data to train on
# - No missing values in any column, which is great - no imputation for nulls
# All features are numeric; categorical variables (SEX, EDucATION, MARRIAGE, PAY_0-PAY_6 are encoded as integers but need to be treated appropriately)
# default.payment.next.month has a mean of 0.221, indicating that about 22% of the customers defaulted on their payment next month
    #This indicates a moderate class imbalance (78% non-default vs 22% default), which we may need to account for (via class weights or sampling)
# EDUCATION ranges from 0-6 (mean of 1.85), but the data dictionary only defines codes 1-4
# MARRIAGE includes a 0 value (mean of 1.55), though only 1-3 are valid
# These 0 and 5,6 codes are "unknown" categories so we should collapse into an "other" bucket.
# PAY_0 through PAY_6 have minnum values of -2 (customer paid 2 months in advance) and max of 8 (customer is 8 months late)
# We should verify that these values align with the documentation, and consider capping or grouping them if extreme delays are rare.
# BILL_AMT columns have some negative minimums, reflecting over-payments or refunds
# BILL_AMT max values exceed 900,000, and PAY_AMT max values exceed 1.6 million - these are extreme outliers
# Medians for BILL_AMT sit around 17k-21k, far below the maximums indicating heavy right skew.
# Will likely need to log-transform or winsorize these amount features, and/or engineer summary stats (e.g evg bill amt) rather than rely on raw values
# AGE spans 21-79 (mean 34), which is a fairly young cohort
# LIMIT_BAL ranges from 10k to 1 million (mean 167k), also highly skewed


# Key takeaways:
# Recode invalid categories in education and marriage
# Flag or transform negative bills (e.g count of negative months)
# handle extreme outliers in BILL_AMT and PAY_AMT (log trransform or summary features)
# Decide how to treat the skew and class imbalance before modeling


# %%
# —— 1. Recode categorical anomalies —— #
# EDUCATION: valid {1,2,3,4}, map {0,5,6} → 4 “others”
df['EDUCATION'] = df['EDUCATION'].replace({0:4, 5:4, 6:4})

# MARRIAGE: valid {1,2,3}, map 0 → 3 “others”
df['MARRIAGE']  = df['MARRIAGE'].replace({0:3})

# convert to category dtype
for col in ['SEX','EDUCATION','MARRIAGE']:
    df[col] = df[col].astype('category')


# —— 2. Rename target & drop ID —— #
df = df.rename(columns={'default.payment.next.month':'default_next_month'})
df = df.drop(columns=['ID'], errors='ignore')


# —— 3. Remove duplicates —— #
dups = df.duplicated().sum()
print(f"Duplicate rows: {dups}")
if dups > 0:
    df = df.drop_duplicates()


# —— 4. Flag negative bill months —— #
bill_cols = [f'BILL_AMT{i}' for i in range(1,7)]
df['num_neg_bill_months'] = (df[bill_cols] < 0).sum(axis=1)


# —— 5. Winsorize extreme amounts —— #
# Clip the 1st & 99th percentile to reduce impact of massive outliers
amt_cols = bill_cols + [f'PAY_AMT{i}' for i in range(1,7)]
for col in amt_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)


# —— 6. Cap PAY delay codes —— #
# Actual PAY_* columns are PAY_0 and PAY_2 through PAY_6
pay_cols = ['PAY_0'] + [f'PAY_{i}' for i in range(2,7)]
for col in pay_cols:
    df[col] = df[col].clip(-2, 8)


# Quick check
print(df[['EDUCATION','MARRIAGE','num_neg_bill_months'] + amt_cols[:3]].describe())

# %%
print('ID' in df.columns)  # should print False
print(df.columns)          # will list only the remaining feature names


# %%
# EDA VISUALS

# 1. Histogram of num_neg_bill_months by default status
default = df[df['default_next_month'] == 1]['num_neg_bill_months']
non_default = df[df['default_next_month'] == 0]['num_neg_bill_months']
bins = range(df['num_neg_bill_months'].min(), df['num_neg_bill_months'].max() + 2)
plt.hist([non_default, default], bins=bins, label=['Non-default', 'Default'], alpha=0.7)
plt.legend()
plt.title('Distribution of Number of Negative Bill Months by Default Status')
plt.xlabel('Number of Negative Bill Months')
plt.ylabel('Count')
plt.show()

# 2. Boxplot of LIMIT_BAL by default status
limit_non_def = df[df['default_next_month'] == 0]['LIMIT_BAL']
limit_def = df[df['default_next_month'] == 1]['LIMIT_BAL']
plt.boxplot([limit_non_def, limit_def], labels=['Non-default', 'Default'])
plt.title('Credit Limit Distribution by Default Status')
plt.ylabel('LIMIT_BAL')
plt.show()

# 3. Bar chart of feature correlation with default
corr = df.corr()['default_next_month'].drop('default_next_month').sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(corr.index, corr.values)
plt.xticks(rotation=90)
plt.title('Feature Correlation with Default')
plt.ylabel('Correlation Coefficient')
plt.tight_layout()
plt.show()


# %%
## *Figure 2 – Correlation of Engineered Features with Default**  

# Plot top 7 engineered features by Pearson correlation
eng_feats = [
    'num_neg_bill_months','utilization','max_delay',
    'avg_delay','payment_ratio','ever_late','bill_trend'
]
corr_vals = (
    df[eng_feats + ['default_next_month']]
      .corr()['default_next_month']
      .drop('default_next_month')
      .sort_values(ascending=False)
)
# Display bar chart
corr_vals.head(7).plot.bar(
    title='Corr of Engineered Features with Default',
    ylabel='Pearson r',
    figsize=(6,3)
)

# %%
# 1. Grab only the numeric columns + target
numeric_feats = df.select_dtypes(include='number').columns.drop('default_next_month')
corr_all = df[numeric_feats.to_list() + ['default_next_month']] \
              .corr()['default_next_month'] \
              .drop('default_next_month')

# 2. Take the 3 strongest positive and 3 strongest negative
top3 = corr_all.nlargest(3)
bot3 = corr_all.nsmallest(3)
combined = pd.concat([top3, bot3])

# 3. Plot
plt.figure(figsize=(6,3))
colors = ['tab:blue' if v>0 else 'tab:red' for v in combined.values]
combined.plot.bar(color=colors)
plt.title('Top & Bottom Feature Correlations with Default')
plt.ylabel('Pearson r')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# KEY TAKEAWAYS FROM EDA AND HOW THEY WILL GUIDE NEXT STEPS

# Over-Payment Behaviour (num_neg_bill_months)
# - Customers who defaulted had a higher average of negative bill months (1.5) compared to non-defaults (0.8)
    # What we see: Almost everyone has zero negative-bill months, very few clients overpaid even once
    # Default signal: dafaulters are less likely to have any months of over-payment
    # Implication: Over-payment (a sign of extra cushion or discipline) is protective, so keep num_neg_bill_months as a feature

# Credit Limit (LIMIT_BAL)
    # What we saw: Those who default tnd to have lower credit limits. The median limit for defaulters sits noticeably below that of non-defaulters.
    # Implication: Credit limit is inversely related to default risk. We'll use it directly and also build utilization ratio (e.g. LIMIT_BAL / avg bill) to capture how close someone is to maxing out their line

# Repayment Delays (PAY_0-PAY_6)
    # What we saw: The PAY_0 (most recent delay) has the strongest positive correlation (.32) with default, followed by PAY_2, PAY_3, etc.
    # Implication: Raw delay codes are very predictive. In addition to including each month's code we should also engineer:
        # A summary of max delay or avg delay
        # Buckets (e.g. on time vs short delay vs long delay)
        # A simple flaf ("ever late?)

# Bill & Payment Amounts
    # What we saw: After winsorization, the six BILL_AMT and PAY_AMT featres individually show only modest correlationwith default.
    # Implication: Rather than feeding in six seperate, highly skewed amounts into the odel, we'll create:
        # Aggregate features: avg bill, avg payment
        # Ratios: payment / bill, utilization (LIMIT_BAL / avg bill)
        # Trends: e.g. the month-to-month change in bill amount

# With these features we'll capture the strongest signals - rpayment behaviour, credit utilization, and over-payments - in a compact form that's ready for model training.

# %%
# Define the operations and their impacts
ops = [
    ("Rows before cleaning",           orig_df.shape[0]),
    ("Rows after cleaning",            df.shape[0]),
    ("Invalid EDUCATION recoded",      orig_df['EDUCATION'].isin([0,5,6]).sum()),
    ("Invalid MARRIAGE recoded",       orig_df['MARRIAGE'].isin([0]).sum()),
    ("Duplicates dropped",             orig_df.duplicated().sum()),
    ("Negative‐bill months flagged",   (df[[f'BILL_AMT{i}' for i in range(1,7)]] < 0).sum().sum()),
    ("Winsorization applied",          "1st–99th percentile clip")
]

# Turn into a DataFrame and display
table1 = pd.DataFrame(ops, columns=['Operation', 'Value'])
display(table1)

# %%
####################################################### FEATURE ENGINEERING #####################################################################

# %%
# Define columns
bill_cols      = [f'BILL_AMT{i}'   for i in range(1, 7)]
pay_amt_cols   = [f'PAY_AMT{i}'    for i in range(1, 7)]
pay_delay_cols = ['PAY_0'] + [f'PAY_{i}' for i in range(2, 7)]

# 1. Aggregate Features
df['avg_bill'] = df[bill_cols].mean(axis=1)
df['avg_pay']  = df[pay_amt_cols].mean(axis=1)

# 2. Ratio Features
df['utilization']   = df['LIMIT_BAL'] / df['avg_bill']
df['payment_ratio'] = df['avg_pay'] / df['avg_bill']

# 3. Delay Summaries
df['max_delay'] = df[pay_delay_cols].max(axis=1)
df['avg_delay'] = df[pay_delay_cols].mean(axis=1)
df['ever_late'] = (df[pay_delay_cols] > 0).any(axis=1).astype(int)

# 4. Trend Features
df['bill_trend'] = df['BILL_AMT6'] - df['BILL_AMT1']
df['pay_trend']  = df['PAY_AMT6']  - df['PAY_AMT1']

# 5. Optional: Delay Buckets
df['delay_bucket'] = pd.cut(
    df['max_delay'],
    bins=[-1, 0, 1, 3, 8],
    labels=['on_time', 'short_delay', 'medium_delay', 'long_delay']
).astype('category')

# 6. Quick check of all new features
print(df[['avg_bill','avg_pay','utilization','payment_ratio',
          'max_delay','avg_delay','ever_late',
          'bill_trend','pay_trend','delay_bucket']].describe(include='all'))
print("\nDelay bucket counts:")
print(df['delay_bucket'].value_counts())



# %%
# The summary output reveals two things we need to clean up before moving on
# 1. Infinite / NaN in utilization & payment_ratio
    # Those infinites come from dividing by avg_bill values that are zero or negative (which happens if a client over-paid more than they owed on average).
    # To fix this let's floor avg_bill at a small positive value before computing ratios.

# 2. Missing buckets in delay_bucket
    # We saw only 24,783 of 29,965 rows got a bucket because max_delay == -2 fell outside of my bins. We should incude those negative delays in the "on_time" group

# %%
# —— Post‐processing of engineered features —— #

# 1. Fix avg_bill for ratio calculations
#    Floor at 1 (or at the smallest positive avg_bill)
min_pos = df.loc[df['avg_bill'] > 0, 'avg_bill'].min()
df['avg_bill_adj'] = df['avg_bill'].clip(lower=min_pos)

# Recompute ratios on the adjusted base
df['utilization']   = df['LIMIT_BAL'] / df['avg_bill_adj']
df['payment_ratio'] = df['avg_pay']    / df['avg_bill_adj']

# 2. Re‐define delay buckets to include negative delays
df['delay_bucket'] = pd.cut(
    df['max_delay'],
    bins=[-3, 0, 1, 3, 8],            # now covers -2 in the first bin
    labels=['on_time', 'short_delay', 'medium_delay', 'long_delay']
).astype('category')

# 3. Quick sanity‐check
print("Utilization/p_ratio finite? ", 
      np.isfinite(df[['utilization','payment_ratio']]).all().all())
print("\nDelay bucket counts:\n", df['delay_bucket'].value_counts(dropna=False))


# %%
# """ # FEATURE ENGINEERING SUMMARY:

# Over‐Payment Flag
   # Retained num_neg_bill_months from cleaning to capture how many months a client over‐paid (protective signal).

# Aggregate Amounts
    # avg_bill = mean of BILL_AMT1…6
    # avg_pay = mean of PAY_AMT1…6

# Adjusted Base for Ratios
    # Floored avg_bill to the smallest positive value (avg_bill_adj) to avoid division by zero/negatives.

# Ratio Features
    # utilization = LIMIT_BAL ÷ avg_bill_adj (credit‐usage intensity)
    # payment_ratio = avg_pay ÷ avg_bill_adj (how fully bills are paid)

# Repayment Delay Summaries
   # max_delay = max of PAY_0,PAY_2…6 (worst‐month delay)
   # avg_delay = mean of those same delay codes
   # ever_late = 1 if any PAY_* > 0 else 0

# Delay Buckets
   # Binned max_delay into four categories—on_time, short_delay, medium_delay, long_delay—ensuring negative delays (early payments) map to “on_time.”

# Trend Features
   # bill_trend = BILL_AMT6 − BILL_AMT1 (change in outstanding balance)
   # pay_trend = PAY_AMT6 − PAY_AMT1 (change in monthly payment)

# These steps compactly encode payment behavior, credit utilization, and repayment discipline—key drivers of default risk—into a set of model‐ready features.

# %%
########################################## TRAIN/TEST SPLIT & BASELINE MODELING ##########################################

# %%
# 1. Split the data
X = df.drop('default_next_month', axis=1)
y = df['default_next_month']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# 2. Identify feature groups
categorical_feats = ['SEX', 'EDUCATION', 'MARRIAGE', 'ever_late', 'delay_bucket']
numeric_feats     = [c for c in X.columns if c not in categorical_feats]

# 3. Build preprocessing pipelines
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', categorical_transformer, categorical_feats)
])

# 4a. Logistic Regression pipeline
pipe_lr = Pipeline([
    ('preproc', preprocessor),
    ('clf', LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    ))
])

# 4b. Random Forest pipeline
pipe_rf = Pipeline([
    ('preproc', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    ))
])

# 5. Fit both models
pipe_lr.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

# 6. Evaluate on the test set
for name, model in [('Logistic Regression', pipe_lr), ('Random Forest', pipe_rf)]:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print(f"\n{name} — Classification Report")
    print(classification_report(y_test, y_pred))
    print(f"{name} — ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")


# %%
# INTERPRETATION OF BASE MODEL RESULTS

# Metric / Class	                    Logistic Regression	        Random Forest
# ROC-AUC	                                    0.755	               0.754

# Default = 1		
# • Precision	                                0.45	                0.63
# • Recall	                                    0.62	                0.34
# • F1-Score	                                0.52	                0.44

# Non-Default = 0		
# • Precision	                                0.88	                0.83
# • Recall	                                    0.79	                0.94
# • F1-Score	                                0.83	                0.89

# Logistic Regression catches more of the true defaulters (Recall 62%) at the cost of more false alarms (Precision 45%).
# Random Forest has higher precision on defaulters (63%) but misses most of them (Recall 34%).
# Both models have very similar overall discrimination (ROC-AUC ~0.75).

# TRADEOFF?

# If missing a defaulter is more costly (e.g. you’d rather investigate more false positives than overlook a real default), Logistic Regression is your better baseline.

# If minimizing false alarms is critical (e.g. you don’t want to waste resources on non-defaulters), you might prefer Random Forest or tune its threshold.



# %%
######################################### HYPERPARAMETER TUNING ##########################################

# %%
# Let's use GridSearchCV (or RandomizedSearchCV) to find the best hyperparameters for our models.
    # Logistic Regression: C (inverse regularization), penalty (l1 vs l2), solver.
    # Random Forest: n_estimators, max_depth, min_samples_split, max_features.

# —— A. Logistic Regression Tuning —— #
param_grid_lr = {
    'clf__C':        [0.01, 0.1, 1, 10, 100],
    'clf__penalty':  ['l1', 'l2'],
    'clf__solver':   ['liblinear']                 # liblinear supports both l1 and l2
}

grid_lr = GridSearchCV(
    pipe_lr,
    param_grid_lr,
    cv=5,
    scoring='roc_auc',
    #Try recal?
    n_jobs=-1,
    verbose=1
)
grid_lr.fit(X_train, y_train)

print("Best LR params:", grid_lr.best_params_)
print("Best LR AUC:",  grid_lr.best_score_)

# —— B. Random Forest Tuning —— #
param_grid_rf = {
    'clf__n_estimators':    [100, 200],
    'clf__max_depth':       [None, 10, 20],
    'clf__min_samples_leaf':[1, 5],
    'clf__max_features':    ['sqrt', 'log2']
}

grid_rf = GridSearchCV(
    pipe_rf,
    param_grid_rf,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train, y_train)

print("Best RF params:", grid_rf.best_params_)
print("Best RF AUC:",  grid_rf.best_score_)

# %%
# Here’s what the tuning results tell us:

# Logistic Regression

    # Best hyperparameters:

        # C = 10 (relatively low regularization)

        # L1 penalty (driving some coefficients to zero for sparsity)

        # liblinear solver (required for L1)

        # Cross-validated AUC: 0.763

# Takeaway: Tuning boosted LR’s discrimination from ~0.755 to ~0.763 AUC, suggesting that relaxing regularization and using an L1 penalty improves its ability to separate defaulters from non-defaulters.

# Random Forest

    # Best hyperparameters:

    # n_estimators = 200 (more trees than baseline)

    # max_depth = 10 (controls tree complexity)

    # min_samples_leaf = 5 (avoids tiny terminal nodes)

    # max_features = “sqrt” (limits features per split to reduce overfitting)

# Cross-validated AUC: 0.786


# Takeaway: RF saw a larger lift—from ~0.754 to ~0.786 AUC—by tuning depth, leaf size, and number of trees, indicating a stronger ability to capture non-linear interactions in the data.

# Overall: Both models improved with tuning, but the Random Forest now shows superior cross-validated performance. Next, we’ll evaluate these tuned models on the hold-out test set to confirm which truly performs best in practice.


# %%
# —— Evaluate Tuned Models on Test Set —— #

from sklearn.metrics import classification_report, roc_auc_score

# 1. Retrieve best estimators
best_lr = grid_lr.best_estimator_
best_rf = grid_rf.best_estimator_

# 2. Predict & evaluate
for name, model in [('Tuned Logistic Regression', best_lr),
                    ('Tuned Random Forest',     best_rf)]:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{name} — Classification Report")
    print(classification_report(y_test, y_pred))
    print(f"{name} — ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")


# %%
# INTERPRETATION OF TUNED RESULTS

# Test‐Set Performance Summary

# Metric / Class	        Tuned Logistic Regression	        Tuned Random Forest
# ROC-AUC	                            0.755	                           0.774
# Accuracy	                            0.75	                           0.77

# Default = 1		
# Precision	                            0.45	                           0.49
# Recall	                            0.62	                           0.59
# F1-Score	                            0.52	                           0.54

# Non-Default = 0		
# Precision	                            0.88	                           0.88
# Recall	                            0.79	                           0.83
# F1-Score	                            0.83	                           0.85

# Key Takeaways:
#   Random Forest wins on overall discrimination (AUC 0.774 vs. 0.755) and accuracy (0.77 vs. 0.75).

#   RF also edges out LR on precision and F1 for the default (positive) class, though LR has a slight recall advantage.

#   Given the business priority of flagging genuine defaulters (and the higher AUC/F1), the tuned Random Forest is our best pick.

# %%
# Finding optimal threshold for our RF Model

from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, ConfusionMatrixDisplay

# 1. Get RF predicted probabilities
proba = best_rf.predict_proba(X_test)[:, 1]

# 2. Compute precision, recall & thresholds
precision, recall, thresholds = precision_recall_curve(y_test, proba)

# 3. Find the threshold that maximizes F1
f1_scores   = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
best_idx    = np.nanargmax(f1_scores)
best_thresh = thresholds[best_idx]
print(f"Optimal threshold (max F1): {best_thresh:.3f}")

# 4. Plot the Precision–Recall curve with the optimal point
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label='PR curve')
plt.scatter(recall[best_idx], precision[best_idx], 
            color='red', label=f'Best F1 @ {best_thresh:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve (Tuned RF)')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Apply that threshold and re-evaluate
y_pred_thresh = (proba >= best_thresh).astype(int)
print("\nClassification Report at Optimal Threshold:")
print(classification_report(y_test, y_pred_thresh))

# 6. Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_thresh, normalize='true')
plt.title(f'Confusion Matrix @ thresh={best_thresh:.2f}')
plt.show()


# %%
# INTERPRETATION OF RESULTS

# Optimal Cutoff for F1
    # Threshold = 0.540 (instead of the default 0.5) maximizes your F1-score, giving you the best balance between precision and recall for defaulters.

# Precision–Recall Trade‐Off
    # At thr=0.540:
    # Precision (defaulters) = 0.52
    # Recall (defaulters) = 0.56
    # F1-score = 0.54

# By moving the cutoff up from 0.50 to 0.54, we've increased precision (fewer false positives) at the cost of a small drop in recall. The F1 stays flat at 0.54, but the precision/recall mix is now more even.

# Overall Accuracy & Non‐Default Performance
    # Accuracy rises to 0.79
    # Non‐Default Class: 85% of good accounts are correctly retained, 15% flagged as false positives.

# Confusion Matrix (Normalized)
    # True Negatives (non-default correctly predicted): 85%
    # False Positives: 15%
    # False Negatives (missed defaulters): 44%
    # True Positives: 56%



# %%
# FOR OUR LOGISTIC REGRESSION MODEL

# 1. Get LR predicted probabilities
proba_lr = best_lr.predict_proba(X_test)[:, 1]

# 2. Compute precision, recall & thresholds
precision, recall, thresholds = precision_recall_curve(y_test, proba_lr)

# 3. Compute F1 for each threshold & find the best
f1_scores   = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
best_idx_lr = np.nanargmax(f1_scores)
best_thresh_lr = thresholds[best_idx_lr]
print(f"LR optimal threshold (max F1): {best_thresh_lr:.3f}")

# 4. Plot the PR curve with the optimal point
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label='LR PR curve')
plt.scatter(recall[best_idx_lr], precision[best_idx_lr],
            color='red', label=f'Best F1 @ {best_thresh_lr:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve (Tuned LR)')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Apply the threshold and re‐evaluate
y_pred_lr_thresh = (proba_lr >= best_thresh_lr).astype(int)
print("\nClassification Report at Optimal LR Threshold:")
print(classification_report(y_test, y_pred_lr_thresh))

# 6. Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr_thresh, normalize='true')
plt.title(f'LR Confusion Matrix @ thresh={best_thresh_lr:.2f}')
plt.show()


# %%
# INTERPRETATION OF LR RESULTS
# 1. Optimal Cutoff for F1
    # Threshold = 0.602 (instead of 0.5) maximizes Logistic Regression’s F1-score.

# 2. Precision–Recall Trade-Off at thr=0.602
    # Precision (defaulters) = 0.49
    # Recall (defaulters) = 0.57
    # F1-score = 0.53

# Compared to the default 0.5 cutoff, you’ve slightly increased precision at the expense of a bit of recall, balancing both at an F1 of ~0.53.

# 3. Overall Performance
    # Accuracy = 0.77
    # Non-Default Class
    # True Negative Rate = 83%
    # False Positive Rate = 17%

# 4. Comparing to Random Forest
    # RF (at its optimal thr=0.54) achieved F1 = 0.54 with precision 0.52 & recall 0.56.
    # LR’s best F1 = 0.53 with precision 0.49 & recall 0.57.
    # RF still holds a slight edge in balanced F1 and precision, while LR edges recall by a hair.

# %%
# WHY WE ARE CHOOSING RANDOM FOREST:
# Best Discrimination: RF achieve the highest cross-validated AUC(0.786) and test-set AUC (0.774), beating LR
# Superior Balance: At its F1-optimal cutoff (0.54),RF delivers the strongest combination of precision (52%) and recall (56%) for detecting defaulters.
# Handles non-linearitis: Its tree-based structure captures complex interactions (e.g. between repayment delays and utilization) without extensive manual feature crossing.

# %%
# NEXT STEPS: We want to target high recall with controlled precision on our RF model.
# This means we will:
#                       - Decide on the minimum proportion of defaulters we must catch (lets go with >= 70% recall first). High recall minimizes costly missed defaults
#                       - Find the corresponding threshold to locate the smallest probability cutoff that achieves our recall target. Ensures I am only tightening the bar as much as neccessary
#                       - Evaluation precision at that recall by running a classification report at the new cutoff to see resulting precision. If precision remains acceptabble, lock this threshold in
#                       - Iterate as needed. If recall comfortably exceeps my goal with good precision I can rais the bar (higher recall target) to reduce false positives further
#                       - Finalize & Deploy: Once I have a threshold that meets my recall target with acceptable precision, I can deploy the tuned Random Forest model with this cutoff for production use.

# %%
# Here we will look for an actual recall at atleast .70% with the tuned Random Forest model, and evaluate the resulting precision at that threshold.



# %% [markdown]
# 

# %%
# 1. Get RF probabilities
proba = best_rf.predict_proba(X_test)[:, 1]

# 2. Compute Precision–Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, proba)

# 3. Align recall[1:] ↔ thresholds
rec_for_thr = recall[1:]  # length == len(thresholds)

# 4. Identify all thresholds with recall ≥ target
target_recall = 0.70
idxs = np.where(rec_for_thr >= target_recall)[0]

if len(idxs) == 0:
    # no threshold meets the target; fall back to minimum threshold
    best_idx = 0
    print(f"No threshold yields recall ≥ {target_recall:.2f}, using lowest cutoff.")
else:
    # pick the largest threshold that still meets recall ≥ target
    best_idx = idxs[-1]

best_thresh = thresholds[best_idx]
print(f"Chosen threshold for ≥{int(target_recall*100)}% recall: {best_thresh:.3f}")

# 5. Evaluate at this threshold
y_pred = (proba >= best_thresh).astype(int)
print("\nClassification Report @70% recall:")
print(classification_report(y_test, y_pred))

# 6. Plot PR curve with chosen point
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label='PR curve')
# recall index in the full array is best_idx+1
plt.scatter(recall[best_idx+1], precision[best_idx+1],
            color='red', label=f'Thr={best_thresh:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve (Tuned RF)')
plt.legend()
plt.tight_layout()
plt.show()

# 7. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, normalize='true'
)
plt.title(f'Confusion Matrix @ thr={best_thresh:.2f}')
plt.show()


# %%
# What the 70% recall threshold means for our model:
# Chosen cutoff: 0.382, Recall (defaulters): 0.70 (by design), Precision: 0.39, F1-score: 0.50, Non-default recall (true negatives: 0.69, false-positive rate: 0.31)
# In plain terms we'll catch 7 out of 10 actual defaulters, but 61% of the accounts I flag willturn out to be non-defaulters.

# Let's iterate on Recall Target

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Get tuned RF probabilities and PR curve
proba = best_rf.predict_proba(X_test)[:, 1]
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, proba)

# Align recall_vals[1:] with thresholds
rec_for_thr = recall_vals[1:]  

# 2. Iterate over recall targets
for target_recall in [0.60, 0.65, 0.70]:
    # Find all thresholds that achieve at least the target recall
    idxs = np.where(rec_for_thr >= target_recall)[0]
    # Choose the largest cutoff that still meets the recall target
    best_idx = idxs[-1] if len(idxs) > 0 else len(thresholds) - 1
    thr = thresholds[best_idx]
    
    # 3. Apply threshold and compute metrics
    y_pred = (proba >= thr).astype(int)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    
    # 4. Print results
    print(f"\n=== Target Recall: {int(target_recall*100)}% ===")
    print(f"Chosen threshold: {thr:.3f}")
    print(f"Precision:           {prec:.3f}")
    print(f"Recall:              {rec:.3f}")
    print(f"F1-score:            {f1:.3f}")
    print(f"False-Positive Rate: {fpr:.3f}")



# %%
# Okay BUSINESS TRADEOFF:
# 60% recall @ 0.479 gives us the best precision/F1 balance (47.5% precision, 53.0% F1) with an acceptable false-alarm rate (18.8% of non-defaulters flagged).
# Raising recall to 65% or 70% raises false positives quickly (25–31% FPR) and drags precision below 43%.

# If the collections team can handle close to 19% false alarms in order to catch 60% of defaulters, let's lock that in.

# %%
# Lets see if we can create a Cost-Based Threshold to put a dollar value on the cost of a missing defult vs a false alarm.
# Maybe we can pick the threshold that minimizes expected cost rather than simply meeting a recall target.

# %%
# Define unit costs (arbitrary 10:1 pilot ratio)
cost_FN = 10   # e.g. $10 cost per missed defaulter
cost_FP = 1    # e.g. $1 cost per false alarm

# Sweep through candidate thresholds
proba = best_rf.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, proba)

costs = []
for thr in thresholds:
    y_pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    exp_cost = fn * cost_FN + fp * cost_FP
    costs.append(exp_cost)

best_idx = np.argmin(costs)
best_thr_cost = thresholds[best_idx]
print("Cost-optimal threshold:", best_thr_cost)


# %%
# At threshold 0.196 we'll flag more accounts as potential defaulters, catching more true positives.
# Precision will fall, but the expected monetary cost in minimized under my 10:1 cost ratio
# Lets see the actual metrics for this one

# %%
thr_cost = 0.1956244934449555
proba = best_rf.predict_proba(X_test)[:, 1]
y_pred_cost = (proba >= thr_cost).astype(int)

# 1. Classification report
print(f"\nClassification Report @ cost-optimal thr={thr_cost:.3f}")
print(classification_report(y_test, y_pred_cost))

# 2. Confusion matrix (normalized)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_cost, normalize='true'
)
plt.title(f'Confusion Matrix @ thr={thr_cost:.3f}')
plt.show()


# %% [markdown]
# 

# %%
# For stakeholders lets extract and plot feature importance (why the model flags someone)
# Math for 0.70/account: comes directly from the confusion‐matrix counts at your cost-optimal cutoff combined with the 10 : 1 cost ratio.

#   From my confusion matrix @ thr=0.196

#       False Positives (FP): 79% of 4 667 non-defaulters ≈ 3 687

#       False Negatives (FN): 3.6% of 1 326 defaulters ≈ 48

#       Applying unit costs:
#           cost_FP = $1 per false alarm
#           cost_FN = $10 per missed default
#           Total Cost=(3687×1)+(48×10)=3687+480=4167
#       Per‐account average:
#                           4167/5993 = $ 0.70 per account

# On average spending about $0.70 in combined outreach and missed-loss cost for each client scored under that threshold.



# %%
# 1. Pull importances out of the RF estimator in your pipeline
imps = best_rf.named_steps['clf'].feature_importances_
# 2. Reconstruct feature names
num_feats = numeric_feats
cat_feats = best_rf.named_steps['preproc'] \
                        .named_transformers_['cat'] \
                        .named_steps['onehot'] \
                        .get_feature_names_out(categorical_feats)
feat_names = np.concatenate([num_feats, cat_feats])

# 3. Build & plot top-10
fi = pd.Series(imps, index=feat_names) \
       .sort_values(ascending=False) \
       .head(10)
fi.plot.barh()
plt.gca().invert_yaxis()
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.show()


# %%
# Top Predictors

    # PAY_0 (most recent repayment delay) and max_delay lead the list, confirming that how late a customer is on their very last statement is the single strongest risk signal.

    # The ever_late_1 flag (whether they were ever late) and the delay buckets (on_time, medium_delay) also rank highly—reinforcing that any history of lateness, and its severity, matter a lot.

    # avg_delay comes next, capturing overall delay behavior, followed by utilization (credit‐usage intensity) and avg_pay.

# Interpretation

    # Repayment behavior dominates: The model most heavily weights recent and aggregated delay measures over dollar‐volume features.

    # Credit usage & payment size play a secondary role—higher utilization and lower payment coverage still contribute meaningfully.



# %%
# —— Summary Table of Threshold Strategies ——
import pandas as pd

data = [
    {'Strategy': 'F1-optimal',        'Threshold': 0.54,  'Precision': 0.52, 'Recall': 0.56, 'F1': 0.54, 'FPR': 0.21},
    {'Strategy': 'Recall-targeted',   'Threshold': 0.479, 'Precision': 0.475,'Recall': 0.60, 'F1': 0.53, 'FPR': 0.188},
    {'Strategy': 'Cost-optimal (10:1)','Threshold': 0.196,'Precision': 0.26, 'Recall': 0.96, 'F1': 0.41, 'FPR': 0.79}
]
df_thresholds = pd.DataFrame(data)
display(df_thresholds)

# %%
for name, model in [('Tuned Logistic Regression', best_lr),
                    ('Tuned Random Forest',     best_rf)]:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n{name} — Classification Report")
    print(classification_report(y_test, y_pred))
    print(f"{name} — ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

# %%
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
for name, model in [('Logistic Regression', best_lr),
                    ('Random Forest',     best_rf)]:
    # compute ROC
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1], [0,1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Tuned Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# %%
# 1. Recompute predictions at the 60%-recall threshold
thr = 0.479
proba = best_rf.predict_proba(X_test)[:, 1]
y_pred_60 = (proba >= thr).astype(int)

# 2. Classification report (optional)
print(f"Classification Report @ thr={thr:.3f}")
print(classification_report(y_test, y_pred_60))

# 3. Confusion matrix plot
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred_60, 
    normalize='true',   # shows rates instead of counts
    display_labels=['Non-Default','Default']
)
plt.title(f'Confusion Matrix @ thr={thr:.3f}')
plt.show()


# %%




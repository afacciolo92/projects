# %%
# Loading necessary libraries
# pd for data manipulation
# np for numerical operations
# plt for visualization
# sns for statistical plots

import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns in DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import os
from pathlib import Path

# %%
merged_monthly_data = {}

# %%
# Define the directory where my raw files are located
raw_data_dir = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data"


# %%
# loading the datasets
# low memory is set to False to avoid dtype warnings and added ensures all columns are typed consistently
# To avoid memory issues, I have set nrows to 5000 for the calendar sample
# Only taking the first month of each datasset to structure a cleaning blueprint for each

listings = pd.read_csv("listings.csv.gz", low_memory=False)
neighbourhoods = pd.read_csv("neighbourhoods.csv")
calendar_sample = pd.read_csv("calendar.csv.gz", nrows=5000)

# %%
# Inspecting the structure of each dataset
# I will use the .info() method to get a concise summary of each DataFrame including the number of non-null entries and data types
print("Listings DataFrame Info:")
print(listings.info())
print("\nCalendar Sample DataFrame Info:")
print(calendar_sample.info())
print("\nNeighbourhoods DataFrame Info:")
print(neighbourhoods.info())

# %%
# Time to view the first few rows of each DataFrame to further understand
print("Listings:")
display(listings.head())

print("\nCalendar Sample:")
display(calendar_sample.head())

print("\nNeighborhoods:")
display(neighbourhoods.head())


# %%
# Big Picture: Understanding the Airbnb Dataset
# The listings dataset is the central listing info, host, price, location, reviews, amenities
# The calendar dataset shows daily availability and price for each listing - helps detect pricing trends & demand
# The neighbourhoods dataset provides geographical context

# %%
# I will begin cleaning the datasets in the following order:
# 1. Listings: this will be the core dataset that I will merge into the others
# 2. Calendar: depends heavily on listings_id from listings and enriches it
# 3. Neighbourhoods: helps group or map listings


# %%
# Cleaning the Listings Dataset

# Cleaning the listings dataset involves:
# 1. Price Column: Problem; it is a string with a dollar sign and commas. Solution: Remove the dollar sign and commas, convert to float for analysis.

listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)

# %%
# 2. host_is_superhost column: Problem; it is a string with 't' and 'f'. Solution: Convert to boolean.
# Easier to use for filtering, grouping, and analysis.

listings['host_is_superhost'] = listings['host_is_superhost'].map({'t': True, 'f': False})

# %%
# 3. host_response_rate column: Problem; it is a string with a percentage sign. Solution: Remove the percentage sign and convert to float.
# This will help in understanding the responsiveness of hosts. Can be used to perform calculations and statistical analysis.

listings['host_response_rate'] = listings['host_response_rate'].str.replace('%', '')
listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'], errors='coerce')  # force NaN if conversion fails

# %%
# 4. Drop high-null columns:

columns_to_drop = ['calendar_updated', 'license'] # calendar_updated literally has no data, license is not useful for analysis and has many nulls
listings.drop(columns=columns_to_drop, inplace=True)

# %%
listings.drop(columns=['neighbourhood'], inplace=True)  # Drop the neighbourhood column as there is a previously cleaned version included


# %%
# 5. Drop any rows with missing key fields: price, room_type, host_id 
# If a row is missing price or room_type or host_id, it is not useful for analysis.
# Price is essential for any financial analysis, and host_id is necessary for linking to host information.
# room_type was accidently dropped but only the missing values were dropped (there were no missing values in the original dataset. All safe)
listings.dropna(subset=['price', 'room_type', 'host_id'], inplace=True)


# %%
# 6a: Contextual median fill for room attributes based on bedroom count
# I need to run prep code because groupby will require numeric input. The following line ensures all three columns are numeric, and forces any "NaN, "None", or "two" become proper NaN to be filled later
# Step 6b depends on this:
for col in ['bathrooms', 'bedrooms', 'beds']:
    listings[col] = pd.to_numeric(listings[col], errors = 'coerce')


# %%
# 6b. Fill 'bathrooms and 'beds' by groped median on 'bedrooms'
# Missing values in 'bathrooms' and 'beds' can be filled with the median of the respective group of 'bedrooms'.
# This is a contextual fill, as it uses the median of the group to fill in missing values.
for target_col in ['bathrooms', 'beds']:
    listings[target_col] = listings.groupby('bedrooms')[target_col].transform(lambda x: x.fillna(x.median()))

# %%
# 6c. Convert any remaining nulls with global median (just in case)
# this is catch and fill for any remaining NaNs that were not filled in the contextual fill
# In looking at ways to structure the code contextually, I realized that the median fill is a good catch-all for any remaining NaNs

for col in ['bathrooms', 'bedrooms', 'beds']:
    listings[col] = listings[col].fillna(listings[col].median())

# %%
# 7. Convert additional 't' and 'f' columns to boolean
# host_identity_verified could be useful for trust analysis and might affect the price, or guest preference. Will see if I will use in analysis
# instant_bookable reflects host availability and may correlate with higher prices, booking rates. May also be useful for analysis. Fix now may use later.

bool_cols = ['host_identity_verified', 'instant_bookable']
for col in bool_cols:
    listings[col] = listings[col].map({'t': True, 'f': False})

# %%
# Done with listings cleaning
# Let's save the newly cleaned listings data and preview
cleaned_listings = listings.copy()

# %%
cleaned_listings.head()

# %%
# I will now check my cleaned list for remaining NaN's (Ranked list of columns with missing data)

cleaned_listings.isnull().sum().sort_values(ascending=False)


# %%
# Not enough insight abovev. I will put code to show ALL columns with missing values and their NaN counts, in descending order
# I will create a new dataframe of missing value counts
null_counts = cleaned_listings.isnull().sum().reset_index()
null_counts.columns = ['Column', 'Missing Values']
null_counts = null_counts[null_counts['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)
null_counts


# %%
# The following columns will be left alone. These are text or descriptive fields or cosmetic in nature - I will not be using to analyze:
# neighbourhood_overview, host_about, description. host_names, host_thumbnail_url, host_picture_url, host_verifications, host_has_profile_pic, host_identity_verified, bathrooms_text

# %%
# After looking at the above "missing value" dataframe I have decided to further clean a few columns.
# Specifically cleaning host_response_rate, host__response_time, host_acceptance_rate, host_is_super_host
# These could be useful if I decide to filter trusted listening, host professionalism, or draw conclusions surrounding sentiment type data

# Fill numeric host metrics with medians. First-pass cleanup - keeps listings usable for analysis without distortion. Avoids skew from outliers.
for col in ['host_response_rate', 'host_acceptance_rate']:
    cleaned_listings[col] = pd.to_numeric(cleaned_listings[col], errors='coerce')
    cleaned_listings[col] = cleaned_listings[col].fillna(cleaned_listings[col].median())

# Fill host_response_time (text) with most common value. Mode is statistically common and safe for categorical text fields
cleaned_listings['host_response_time'] = cleaned_listings['host_response_time'].fillna(
    cleaned_listings['host_response_time'].mode()[0])

# Fill host_is_superhost with False (conservative assumption) There is only 338 missing values here.
cleaned_listings['host_is_superhost'] = cleaned_listings['host_is_superhost'].fillna(False)

# %%
# Verify no missing values remain in those columns
cleaned_listings[['host_response_rate', 'host_acceptance_rate', 'host_response_time', 'host_is_superhost']].isnull().sum()

# %%
# It looks like I did not strip the text (%) from host_acceptance_rate in initial clean.
# Fix: # Fix: Clean and fill host_acceptance_rate properly
# Ensure it's a string before attempting .str.replace
cleaned_listings['host_acceptance_rate'] = cleaned_listings['host_acceptance_rate'].astype(str)
cleaned_listings['host_acceptance_rate'] = cleaned_listings['host_acceptance_rate'].str.replace('%', '', regex=False)

# Convert to numeric
cleaned_listings['host_acceptance_rate'] = pd.to_numeric(cleaned_listings['host_acceptance_rate'], errors='coerce')

# Fill remaining NaNs with median
cleaned_listings['host_acceptance_rate'] = cleaned_listings['host_acceptance_rate'].fillna(
    cleaned_listings['host_acceptance_rate'].median()
)

# %%
# Ensure host_acceptance_rate in cleaned fully
cleaned_listings['host_acceptance_rate'].isnull().sum()

# %%
# I have decided to drop the host_acceptance_rate fully. I am having issues and may not be vital to trust/sentiment signal.
cleaned_listings.drop(columns=['host_acceptance_rate'], inplace=True)

# %%
# Confirm there are no critical NaNs left
cleaned_listings.isnull().sum().sort_values(ascending=False).head(20)

# %%
# See first few rows
cleaned_listings.head()

# View summary of column types and nulls
cleaned_listings.info()


# %%
# I noticed the indexing is off. Must reset Index for clarity
cleaned_listings.reset_index(drop=True, inplace=True)


# %%
# See first few rows
cleaned_listings.head()

# View summary of column types and nulls
cleaned_listings.info()

# Critical Columns fully intact. Price non-null float cleaned and usable, room_type non-null object ready for grouping or filtering
# beds, bathrooms, bedroom non-null float cleaned and filled
# host_is_superhost, instant_bookable non-null booleans ready for any trust or booking analysis
# lattitude, longitude, neighbourhood cleaned non-null ready for geo/spatial work

# %%
# Adding a date_month column to cleaned May 2025 listings dataset
# Purpose: We want to track which month each row of data came from and help analyze trends over time

cleaned_listings['data_month'] = '2025-05'

# %%
cleaned_listings[['id', 'price', 'data_month']].head()

# %%
# I will now clean calendar for May
calendar = pd.read_csv('calendar.csv.gz')

# %%
# Inspecting raw calendar data
print("Shape of calendar data:", calendar.shape)
print("\nColumns:\n", calendar.columns)
print("\nSample rows:\n", calendar.head())
print("\nMissing values:\n", calendar.isnull().sum())
calendar.info()

# %%
# Okay some things I have noticed include a ton of missing values for adjusted price, price is an object and should be a float
# Let's make a copy of the raw calendar
cleaned_calendar = calendar.copy()

# %%
# Dropping adusted price since most is missing
cleaned_calendar.drop(columns=['adjusted_price'], inplace=True)

# %%
#Clean the price column by removing dollar signs and commas and convert to float
cleaned_calendar['price'] = (
    cleaned_calendar['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
)

# %%
# Let's fill missing values in minimum and maximum nights. I will use the global median as the fraction of missing values here are very low
cleaned_calendar['minimum_nights'] = cleaned_calendar['minimum_nights'].fillna(
    cleaned_calendar['minimum_nights'].median()
)
cleaned_calendar['maximum_nights'] = cleaned_calendar['maximum_nights'].fillna(
    cleaned_calendar['maximum_nights'].median()
)


# %%
#check
cleaned_calendar.isnull().sum().sort_values(ascending=False)

# %%
# add the date_month column
cleaned_calendar['data_month'] = '2025-05'

# %%
cleaned_calendar.info()
cleaned_calendar.head()

# %%
# Move on to neighbourhoods
neighbourhoods.info()
neighbourhoods.head()
neighbourhoods.isnull().sum()

# %%
# Drop duplicates if there are any
neighbourhoods.drop_duplicates(inplace=True)

# %%
neighbourhoods['data_month'] = '2025-05'

# %%
cleaned_neighbourhoods = neighbourhoods.copy()
cleaned_neighbourhoods.head()

# %%
# Id like to compare neighbourhood columns in listings with neighbourhoods in neighbourhood to ensure consistency and easy merge

# First, standardize both for comparison
neighbourhoods['neighbourhood'] = neighbourhoods['neighbourhood'].str.strip().str.lower()
listings['neighbourhood_cleansed'] = listings['neighbourhood_cleansed'].str.strip().str.lower()

# Check unique values in each
neighbourhood_unique = set(neighbourhoods['neighbourhood'].unique())
listings_unique = set(listings['neighbourhood_cleansed'].unique())

# Identify mismatches
neighbourhoods_not_in_listings = neighbourhood_unique - listings_unique
listings_not_in_neighbourhoods = listings_unique - neighbourhood_unique

print("Neighbourhoods in 'neighbourhoods.csv' but not in listings:")
print(neighbourhoods_not_in_listings)

print("\nNeighbourhoods in listings but not in 'neighbourhoods.csv':")
print(listings_not_in_neighbourhoods)

# %%
# it looks lke the only mismatch is that a few neighbourhoods from the neighbourhood.csv fileare not present in the listings data for May
# no neighbourhoods from the listings file are missing in the neighbourhoods.csv meaning every listing can be matched to a neighbourhood group if merge on neighbourhood
# Proceed with merge
# Rename 'neighbourhood' to 'neighbourhood_cleansed' in the cleaned_neighbourhoods dataframe
# to match the column name in listings and allow for accurate merging

neighbourhoods_for_merge = cleaned_neighbourhoods.rename(columns={'neighbourhood': 'neighbourhood_cleansed'})

# Merge cleaned listings (with date_month already added) and cleaned neighbourhoods
# using a left join on 'neighbourhood_cleansed'
merged_listings_neigh = cleaned_listings.merge(
    neighbourhoods_for_merge,
    on='neighbourhood_cleansed',
    how='left',
    suffixes=('', '_neigh')  # Prevents accidental column overwrites
)


# %%
#Confirm new column was added
print("New columns added from neighbourhoods data:")
print(set(merged_listings_neigh.columns) - set(cleaned_listings.columns))

# %%
# Check
merged_listings_neigh.info()
merged_listings_neigh.head()

# %%
# While trying to merge I am noticing kernal crashed. I will try and slim down the listings  by removing unneccessary columns
# descrption, neighbourhood_overview, host_about,picture_url, host_thumbnail, host picture

columns_to_drop = [
    'description', 'neighborhood_overview', 'host_about',
    'picture_url', 'host_thumbnail_url', 'host_picture_url'
]
slimmed_listings_neigh = merged_listings_neigh.drop(columns=columns_to_drop, errors='ignore')




# %%
slimmed_listings_neigh.head()

# %%
# I need to chunk the calendar in so my kernal doesn't crash
# I will also filter the calendar to only include listing_ids present in the listings + neighbourhood data
# This is effectively removing data but ensures that I can create solid hypothesis' from a readable merged dataset
filtered_calendar = cleaned_calendar[cleaned_calendar['listing_id'].isin(slimmed_listings_neigh['id'])]


# %%
#chunk the filtered calendar into 3
chunks = np.array_split(filtered_calendar, 3)
merged_chunks = []

# %%
#Merge each chunk with listings+neighbourhoods and collect results
for i, chunk in enumerate(chunks):
    print(f"Merging chunk {i+1} of {len(chunks)}...")
    merged = pd.merge(
        chunk,
        slimmed_listings_neigh,
        how='left',
        left_on='listing_id',
        right_on='id'
    )
    merged['data_month'] = '2025-05'  # Add a date marker for trend analysis
    merged_chunks.append(merged)


# %%
# Concatenate all merged chunks into one final dataframe for May
merged_may_data = pd.concat(merged_chunks, ignore_index=True)


# %%
merged_may_data.head()
merged_may_data.info()

# %%
merged_may_data.head(10)
merged_may_data.sample(10)


# %%
merged_may_data.isna().sum().sort_values(ascending=False).head(20)


# %%
# Final cleaning required

merged_may_data.rename(columns={
    'price_x': 'calendar_price',    # Price from calendar
    'price_y': 'listing_price'      # Price from listings
}, inplace=True)

# Drop redundant data_month columns (we're keeping only final 'data_month')
merged_may_data.drop(columns=[
    'data_month_x',
    'data_month_y',
    'data_month_neigh'
], inplace=True)

# Just in case I didn't filter mismatches earlier
merged_may_data = merged_may_data[~merged_may_data['listing_price'].isna()]

# Reset index after cleaning
merged_may_data.reset_index(drop=True, inplace=True)

# Sort by listing_id and date for better readability and arrange chronologically
merged_may_data.sort_values(by=['listing_id', 'date'], inplace=True)


# %%
merged_may_data.shape
merged_may_data.info()

# %%
merged_may_data.head(3)
merged_may_data.tail(3)


# %%
#taking a sample. Listing-level info is consistent across dates, calendar-llevel info is changing over time, merged listings and calendar prices exist and look correct
# No NaNs in critical fields like prices or availability for this listing
sample_id = merged_may_data['listing_id'].iloc[0]
merged_may_data[merged_may_data['listing_id'] == sample_id]


# %%
## BATCH PROCESSING AND MERGING OF REMAINING MONTHS: JUNE 2024 - APRIL 2025



# I will now try implementing functions to copy the cleaning strategy from each of the cleaned datasets and then merge
# At this point i have gone to the top of my code to introduce os and pathlib
# Get a list of all months available (excluding May, which is already processed)





# %%
import os
import re

raw_data_dir = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data"


# %%
# Identify all months from filenames
months_available = sorted(set([
    re.search(r'_(\d{4}-\d{2})', fname).group(1)
    for fname in os.listdir(raw_data_dir)
    if fname.startswith(("calendar_", "listings_", "neighbourhoods_"))
]))

# Exclude May (already processed)
months_to_process = [month for month in months_available if month != "2025-05"]
print("Months to process:", months_to_process)


# %%
def load_month_data(month, raw_data_dir):
    """
    Loads the raw calendar, listings, and neighbourhoods files for a specific month.
    """
    calendar_path = os.path.join(raw_data_dir, f"calendar_{month}.csv.gz")
    listings_path = os.path.join(raw_data_dir, f"listings_{month}.csv.gz")
    neigh_path = os.path.join(raw_data_dir, f"neighbourhoods_{month}.csv")

    calendar = pd.read_csv(calendar_path, low_memory=False)
    listings = pd.read_csv(listings_path, low_memory=False)
    neighbourhoods = pd.read_csv(neigh_path)

    return calendar, listings, neighbourhoods


# %%
def clean_listings(df, month):
    """
    Cleans a single month's listings data using the same logic as May 2025.
    """

    df = df.copy()

    # Convert price to float
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

    # Convert percentages to float
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False)
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')

    # Clean host_is_superhost BEFORE this function (to avoid kernel crash)
    # ensure it's boolean here again just in case:
    df['host_is_superhost'] = df['host_is_superhost'].astype(bool)

    # Drop less useful or redundant columns
    df.drop(columns=['calendar_updated', 'license', 'neighbourhood'], errors='ignore', inplace=True)

    # Drop rows with critical missing values
    df.dropna(subset=['price', 'room_type', 'host_id'], inplace=True)

    # Convert key numeric columns
    for col in ['bathrooms', 'bedrooms', 'beds']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill nulls with contextual medians
    for target_col in ['bathrooms', 'beds']:
        df[target_col] = df.groupby('bedrooms')[target_col].transform(lambda x: x.fillna(x.median()))
    for col in ['bathrooms', 'bedrooms', 'beds']:
        df[col] = df[col].fillna(df[col].median())

    # Convert booleans from 't'/'f'
    for col in ['host_identity_verified', 'instant_bookable']:
        df[col] = df[col].map({'t': True, 'f': False})

    # Remove host_acceptance_rate (after coercing and filling if needed)
    df['host_acceptance_rate'] = df['host_acceptance_rate'].astype(str).str.replace('%', '', regex=False)
    df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'], errors='coerce')
    df.drop(columns=['host_acceptance_rate'], inplace=True, errors='ignore')

    # Fill in mode for host_response_time
    if df['host_response_time'].isnull().any():
        df['host_response_time'] = df['host_response_time'].fillna(df['host_response_time'].mode()[0])

    # Add timestamp marker
    df['data_month'] = month

    return df


# %%
def clean_calendar(df, month):
    """
    Cleans a single month's calendar file.
    """
    df = df.copy()

    # Drop adjusted_price if it exists
    df.drop(columns=['adjusted_price'], errors='ignore', inplace=True)

    # Clean price string to float
    df['price'] = (
        df['price'].str.replace('$', '', regex=False)
                   .str.replace(',', '', regex=False)
                   .astype(float)
    )

    # Fill in night limits using global median
    df['minimum_nights'] = df['minimum_nights'].fillna(df['minimum_nights'].median())
    df['maximum_nights'] = df['maximum_nights'].fillna(df['maximum_nights'].median())

    # Add the month
    df['data_month'] = month

    return df

# %%
def clean_neighbourhoods(df, month):
    """
    Cleans neighbourhoods data and standardizes naming.
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df['neighbourhood'] = df['neighbourhood'].str.strip().str.lower()
    df['data_month'] = month
    return df

# %%
def merge_all(calendar, listings, neighbourhoods):
    """
    Merges cleaned calendar, listings, and neighbourhoods data into a single dataframe for one month.
    """
    # Prep listings for merge
    listings['neighbourhood_cleansed'] = listings['neighbourhood_cleansed'].str.strip().str.lower()

    # Merge with neighbourhoods
    neigh_for_merge = neighbourhoods.rename(columns={'neighbourhood': 'neighbourhood_cleansed'})
    merged_listings = listings.merge(neigh_for_merge, on='neighbourhood_cleansed', how='left')

    # Drop high-NaN or unneeded columns
    columns_to_drop = ['description', 'neighborhood_overview', 'host_about',
                       'picture_url', 'host_thumbnail_url', 'host_picture_url']
    slimmed_listings = merged_listings.drop(columns=columns_to_drop, errors='ignore')

    # Filter calendar rows to match listings IDs
    filtered_calendar = calendar[calendar['listing_id'].isin(slimmed_listings['id'])]

    # Merge calendar + listings
    merged = pd.merge(filtered_calendar, slimmed_listings, how='left', left_on='listing_id', right_on='id')

    # Add month metadata (comes from calendar)
    merged['data_month'] = calendar['data_month'].iloc[0]

    # Rename price fields
    merged.rename(columns={
        'price_x': 'calendar_price',
        'price_y': 'listing_price'
    }, inplace=True)

    # Drop any redundant month labels
    merged.drop(columns=[
        'data_month_x', 'data_month_y', 'data_month_neigh'
    ], errors='ignore', inplace=True)

    # Filter out broken rows that didn’t merge listings
    merged = merged[~merged['listing_price'].isna()]

    # Final cleanup
    merged.reset_index(drop=True, inplace=True)
    merged.sort_values(by=['listing_id', 'date'], inplace=True)

    return merged


# %%
# CHANGE the month value before each run

month = '2024-06'  # Example: Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")



# %%
month = '2024-07'  # Example: Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2024-08'  # Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2024-09'  # Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2024-10'  # Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2024-11'  # Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2024-12'  # Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2025-01'  # Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2025-02'  # Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2025-03'  # Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
month = '2025-04'  #Next month to process

try:
    print(f"\n Processing {month}...")

    # Load raw files
    calendar, listings, neighbourhoods = load_month_data(month, raw_data_dir)

    # Clean each
    cleaned_listings = clean_listings(listings, month)
    cleaned_calendar = clean_calendar(calendar, month)
    cleaned_neigh = clean_neighbourhoods(neighbourhoods, month)

    # Merge
    merged_month_df = merge_all(cleaned_calendar, cleaned_listings, cleaned_neigh)

    print(f" {month} merged shape: {merged_month_df.shape}")

    #  Save immediately to disk to reduce memory pressure
    merged_month_df.to_parquet(f"merged_{month}.parquet", index=False)
    print(f" Saved merged_{month}.parquet")

    #  Clear RAM
    del calendar, listings, neighbourhoods
    del cleaned_listings, cleaned_calendar, cleaned_neigh
    del merged_month_df
    import gc; gc.collect()

except Exception as e:
    print(f" Failed processing {month}: {e}")

# %%
merged_may_data.to_parquet("/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/merged_2025-05.parquet", 
    index=False)

# %%
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gc

# Path to your cleaned monthly parquet files
merged_data_dir = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data"
final_output_path = os.path.join(merged_data_dir, "full_airbnb_dataset_2024_2025.parquet")

# Step 1: List all monthly files to be merged
parquet_files = sorted([f for f in os.listdir(merged_data_dir) if f.endswith(".parquet") and f.startswith("merged_")])

# Step 2: Inspect schema from the first file to standardize all others
sample_df = pd.read_parquet(os.path.join(merged_data_dir, parquet_files[0]))
column_dtypes = sample_df.dtypes.apply(lambda dt: 'Int64' if str(dt) == 'int64' else dt).to_dict()
all_columns = list(column_dtypes.keys())

# Step 3: Write loop to append data into one consistent file
writer = None

for i, f in enumerate(parquet_files):
    file_path = os.path.join(merged_data_dir, f)
    print(f" Processing {f} ({i+1}/{len(parquet_files)})...")

    df = pd.read_parquet(file_path)

    #  Ensure all expected columns exist
    for col in all_columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[all_columns]  # align column order

    #  Enforce consistent types (handles nullable ints safely)
    df = df.astype(column_dtypes)

    table = pa.Table.from_pandas(df)

    if writer is None:
        writer = pq.ParquetWriter(final_output_path, table.schema, compression="snappy")

    writer.write_table(table)

    # 🧹 Free memory
    del df, table
    gc.collect()

#  Finalize and close writer
if writer:
    writer.close()

print(" Final merged dataset created and saved at:", final_output_path)


# %%
import pyarrow.parquet as pq

# Define the path to the full dataset
parquet_path = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/full_airbnb_dataset_2024_2025.parquet"

# Step 1: Read just the first row group
pf = pq.ParquetFile(parquet_path)

# Inspect the number of row groups
print(f" Row groups in file: {pf.num_row_groups}")

# Step 2: Read the first row group (should be a manageable chunk)
table = pf.read_row_group(0, columns=["data_month", "listing_id", "date"])
sample_df = table.to_pandas()

# Preview
print("Sample from first row group:")
print(sample_df.head())



# %%
# Checking if my merged frame works properly

from collections import Counter

# Reopen the parquet file
pf = pq.ParquetFile(parquet_path)

# Initialize month counter
month_counts = Counter()

# Process each row group one at a time
for i in range(pf.num_row_groups):
    print(f" Processing row group {i+1}/{pf.num_row_groups}...")
    table = pf.read_row_group(i, columns=["data_month"])
    df = table.to_pandas()
    month_counts.update(df["data_month"].value_counts().to_dict())

# Display the results
print("\nRow count per month:")
for month, count in sorted(month_counts.items()):
    print(f"{month}: {count:,}")


# %%
# Making sure all months are properly depicted in dataframe

import pyarrow.parquet as pq
import pandas as pd
import random

# Path to full merged dataset
parquet_path = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/full_airbnb_dataset_2024_2025.parquet"

# Load the file
pf = pq.ParquetFile(parquet_path)

# Get the schema just to confirm
print(f"Total row groups: {pf.num_row_groups}")

# Dictionary to store 1 sample row per month
monthly_samples = []

# Process each row group one at a time
for i in range(pf.num_row_groups):
    table = pf.read_row_group(i, columns=["data_month", "date", "listing_id", "calendar_price",
                                          "neighbourhood_cleansed", "neighbourhood_group",
                                          "room_type", "listing_price"])
    df = table.to_pandas()

    # Sample 1 random row per unique month in the row group
    for month in df['data_month'].unique():
        subset = df[df['data_month'] == month]
        if not subset.empty:
            monthly_samples.append(subset.sample(1))

# Concatenate all samples
sampled_df = pd.concat(monthly_samples, ignore_index=True)

print(f"\n Sampled 1 row from each month. Total rows: {len(sampled_df)}")
display(sampled_df)


# %%
# Re-confirming my analysis scope:
    # What does Airbnb data from the past year reveal about broader economic trends in New Yok City?
    # The scope includes date frange from June 2024 - May 2025
    # My key metric categories will be prices, availability, room types, revenue estimates, review trend

# %%
# Part 1: Price Trend Analysis
    # I will begin by analyzing average nightly price trends across the 12 months
    # My kernal continues to crash so I will import ONLY data_month and calendar_price columnsto complete this

import pandas as pd

# Load only the required columns
cols_to_load = ["data_month", "calendar_price"]
price_df = pd.read_parquet(
    "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/full_airbnb_dataset_2024_2025.parquet",
    columns=cols_to_load
)

# Group by month and calculate average calendar price
price_trend = price_df.groupby("data_month")["calendar_price"].mean().reset_index()

# Confirm
print(price_trend)


# %%
# Trying to interpret the data based on the above:
    # Mean prices rise sharply from after June through August, peaking at 494 in October. This aligns with peak tourism season in NYC
    # Post holiday dip: You can see it steadily dipping after October with Jan being the lowest mean average in the dataset after June, 2024
    # You see it climb back up after Jan and peak again in May 2025, possbily due to pre-sumer travel and events
    # The dip is drastic following December, likely reflecting that people were traveling during the december holidays

# %%
# Let's visualize it
# This plot supports Key Finding 1 in Executive Summary:
    # Highhlights clear seasonality in average calendar prices and aligns with summer tourism peaks and winter dips

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(price_trend["data_month"], price_trend["calendar_price"], marker="o", linestyle="-")
plt.title("Average Nightly Calendar Price (NYC Airbnb, June 2024 – May 2025)")
plt.xlabel("Month")
plt.ylabel("Average Price (USD)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Let's drill down to neghbourhood-level price trends.
    # This will allow us to uncover microeconomic signals like localized tourism spikes, or pricing volatility in specific boroughs
    # Potentially connect changes in pricing to policy shifts, events, or investment in those areas?
    # Focus on neighbourhoods wih the highest number of listings to ensure trends or robust

# Memory saving will just load columns needed

import pyarrow.parquet as pq
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Define the path to the full dataset
parquet_path = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/full_airbnb_dataset_2024_2025.parquet"

# Step 2: Read only the necessary columns into memory
# We're selecting columns needed for neighborhood-level price trend analysis
columns_needed = ["data_month", "calendar_price", "neighbourhood_cleansed", "listing_id"]
table = pq.read_table(parquet_path, columns=columns_needed)
df_neigh = table.to_pandas()



# %%
# Step 3: Identify the top 5 neighborhoods by number of listings
top_neigh = (
    df_neigh.groupby("neighbourhood_cleansed")["listing_id"]
    .nunique()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

# Step 4: Filter data to only include these top 5 neighborhoods
df_top_neigh = df_neigh[df_neigh["neighbourhood_cleansed"].isin(top_neigh)]

# %%
# Step 5: Group by month and neighborhood to calculate average nightly price
neigh_price_trend = (
    df_top_neigh.groupby(["data_month", "neighbourhood_cleansed"])["calendar_price"]
    .mean()
    .reset_index()
)

# %%
# Step 6: Plot the monthly price trend for each top neighborhood
# This plot supports key finding 2:
    # illistrates mictroeconomic variation in price trends across top neighbourhoods 
    # Used to signal local demand differentiation
# Prices are quite high in luxury area of Midtown compared to areas like Harlem and bedford-stu
# Signals a market gap?
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=neigh_price_trend,
    x="data_month",
    y="calendar_price",
    hue="neighbourhood_cleansed"
)
plt.title("Average Nightly Calendar Price by Neighborhood (Top 5 by Listings)")
plt.xlabel("Month")
plt.ylabel("Price (USD)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# I will now examne occupancy rate trends.
# Occupancy equals real demand.
# Potential insights include:
    # Rising or falling occupancy over time (travel behaviour, affordability)
    # Neighbourhoods with high occupancy but low prices - budget travelers indication
    # High prices + low occupancy - market saturation or overpricing?
# Will it indicate economic confidence and seasonal tourism fluctuations?

import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt

# Path to the full dataset
parquet_path = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/full_airbnb_dataset_2024_2025.parquet"

# Open the Parquet file
pf = pq.ParquetFile(parquet_path)

# We'll collect results month by month across row groups
monthly_occupancy = []


# %%
# Process one row group at a time to avoid memory overload
for i in range(pf.num_row_groups):
    # Only load what's needed: availability + data_month
    table = pf.read_row_group(i, columns=["data_month", "available"])
    df = table.to_pandas()

    # Count bookings and availability by month
    grouped = df.groupby("data_month")["available"].value_counts().unstack(fill_value=0)
    for month, row in grouped.iterrows():
        available = row.get("t", 0)
        booked = row.get("f", 0)
        total = available + booked
        occupancy_rate = (booked / total) * 100 if total > 0 else 0
        monthly_occupancy.append({"data_month": month, "occupancy_rate": occupancy_rate})

# Create DataFrame
occupancy_df = pd.DataFrame(monthly_occupancy).sort_values("data_month")

# %%
# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(occupancy_df["data_month"], occupancy_df["occupancy_rate"], marker="o", linestyle="-", color="teal")
plt.title("Estimated Occupancy Rate by Month (NYC Airbnb, June 2024 – May 2025)")
plt.xlabel("Month")
plt.ylabel("Occupancy Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: View data
display(occupancy_df)


# %%
# The jagged output loks to be due to multiple entries per data_month because groupby("data_month")["available"].value._counts() kept subgroups by row group
# Refine to ensure one occupancy per month - to ensure cleaner and more accurate executive summary visualizaton
# Refined version: consolidate 'available' counts across all row groups to compute clean monthly occupancy rates

import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt

# Path to dataset
parquet_path = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/full_airbnb_dataset_2024_2025.parquet"
pf = pq.ParquetFile(parquet_path)

# Dictionary to store running counts
monthly_counts = {}

# Process each row group
for i in range(pf.num_row_groups):
    df = pf.read_row_group(i, columns=["data_month", "available"]).to_pandas()

    # Count total 't' and 'f' values per data_month
    grouped = df.groupby(["data_month", "available"]).size().unstack(fill_value=0)

    for month in grouped.index:
        if month not in monthly_counts:
            monthly_counts[month] = {"t": 0, "f": 0}
        monthly_counts[month]["t"] += grouped.loc[month].get("t", 0)
        monthly_counts[month]["f"] += grouped.loc[month].get("f", 0)

# Compute occupancy rate
monthly_occupancy = []
for month, counts in monthly_counts.items():
    total = counts["t"] + counts["f"]
    rate = (counts["f"] / total) * 100 if total else 0
    monthly_occupancy.append({"data_month": month, "occupancy_rate": rate})

# Create DataFrame
occupancy_df = pd.DataFrame(monthly_occupancy).sort_values("data_month")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(occupancy_df["data_month"], occupancy_df["occupancy_rate"], marker="o", linestyle="-", color="teal")
plt.title("Estimated Occupancy Rate by Month (NYC Airbnb, June 2024 – May 2025)")
plt.xlabel("Month")
plt.ylabel("Occupancy Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: View table
display(occupancy_df)


# The plot below supports Key Finding 3:
    # Reveals declining occupancy despite rising prices
    # Market mismatch or overpricing?

# %%
# Can we go deeper by looking at Price vs. Occupancy by neighbourhood?
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load only required columns for efficiency
columns_needed = ["data_month", "neighbourhood_cleansed", "listing_id", "calendar_price", "available"]
parquet_path = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/full_airbnb_dataset_2024_2025.parquet"
pf = pq.ParquetFile(parquet_path)

dfs = []


# %%
for i in range(pf.num_row_groups):
    print(f" Reading row group {i+1}/{pf.num_row_groups}...")
    table = pf.read_row_group(i, columns=columns_needed)
    df = table.to_pandas()
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)


# %%
# Compute occupancy at listing/month level

df["booked"] = df["available"] == "f"
listing_month_summary = df.groupby(["data_month", "neighbourhood_cleansed", "listing_id"], as_index=False).agg({
    "calendar_price": "mean",
    "booked": "mean"  # percentage of days booked
})


# %%
# Now group at neighborhood + month level
neigh_summary = listing_month_summary.groupby(["data_month", "neighbourhood_cleansed"], as_index=False).agg({
    "calendar_price": "mean",
    "booked": "mean"
})
neigh_summary.rename(columns={"booked": "occupancy_rate"}, inplace=True)

# %%
# Plot price vs. occupancy for latest month (e.g., 2025-05)

latest_month = "2025-05"
subset = neigh_summary[neigh_summary["data_month"] == latest_month]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=subset, x="occupancy_rate", y="calendar_price", hue="neighbourhood_cleansed")
plt.title(f"Price vs. Occupancy Rate by Neighborhood ({latest_month})")
plt.xlabel("Occupancy Rate (0–1)")
plt.ylabel("Average Nightly Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot below shows some interesting data about the occupancy rate versus neighbourhood.
# Luxury accomodations prices are quite high with steady low occupancy rate

# %%
# We have data representing may and reveals a spread between occupancy rates and avg nightly prices across NYC neighbourhoods.
# It looks like we have:
    # High price, low occupancy clusters - Midtown, Tribeca, Chelsea sit at top of the price scale but show moderate to low occupancy. Overpriced relative to demand? Serving niche markets?
    # Moderate price, hgh occupancy - bedford-stuyvesant, queens: affordable rates and strong occupancy represent budget concious?
    # Low occupancy: Many neihbourhoods cluster below 0.3 occupancy rates, even with moderate pices: 0versupply? Unattractive locations?


# %%
# Can we take two additional months and run a camparison to outline a little more of a trend?

import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Target months to compare
target_months = ["2025-05", "2025-03", "2024-08"]

# Define path and columns
parquet_path = "/Users/andrewfacciolo/Desktop/Homework/airbnb_raw_data/full_airbnb_dataset_2024_2025.parquet"
columns_needed = ["data_month", "neighbourhood_cleansed", "listing_id", "calendar_price", "available"]

# %%
# Load filtered data from parquet in chunks
pf = pq.ParquetFile(parquet_path)
filtered_chunks = []

for i in range(pf.num_row_groups):
    print(f" Reading row group {i+1}/{pf.num_row_groups}...")
    table = pf.read_row_group(i, columns=columns_needed)
    df = table.to_pandas()
    df = df[df["data_month"].isin(target_months)]
    filtered_chunks.append(df)
    del df, table
    gc.collect()

# Combine only necessary data
df = pd.concat(filtered_chunks, ignore_index=True)
del filtered_chunks
gc.collect()


# %%
# Calculate per-listing price and occupancy
df["booked"] = df["available"] == "f"
listing_month_summary = df.groupby(["data_month", "neighbourhood_cleansed", "listing_id"], as_index=False).agg({
    "calendar_price": "mean",
    "booked": "mean"
})

# Aggregate to neighborhood-month level
neigh_summary = listing_month_summary.groupby(["data_month", "neighbourhood_cleansed"], as_index=False).agg({
    "calendar_price": "mean",
    "booked": "mean"
})
neigh_summary.rename(columns={"booked": "occupancy_rate"}, inplace=True)

# %%
# Plot: One color per month, different marker for each
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=neigh_summary,
    x="occupancy_rate",
    y="calendar_price",
    hue="data_month",
    style="data_month"
)
plt.title("Price vs. Occupancy Rate by Neighborhood (Selected Months)")
plt.xlabel("Occupancy Rate (0–1)")
plt.ylabel("Avg Nightly Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Key takeaways from this plot:
    # 1 High price listings don't guaruntee high occupancy:
        # in all three months majority of listings above $1000/night consistantly exhibit low occupance rates <0.3
        # These are ikely luxury or niche properties demand is less elastic but volume is low
    # 2 Mid range price clusters shift seasonally:
        # August 2024 we see higher overallprics at moderate occupancy (vacation time!)
        # March 2025 listings shift slightly down in price, occupancy remains varied
        # May 2025 prices rebound but occupancy remains dispersed (over saturaton of listings? post-peak travel hesitation despite spring events?)
    # 3 Neighbourhood level heterogeneity:
        # Clister along the bottom (high occupancy, low price) is tighter in March
        # In May, more listings re-enter high-price territory withot proportional occupancy increases
        # Suggests localized demand weakness or pricing misalignmnet

# %%




import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# Convert AgeUponOutcome to years
def convert_age_to_years(age_str):
    if pd.isna(age_str):
        return None

    age_str = age_str.lower().strip()
    parts = age_str.split()
    if len(parts) != 2:
        return None  

    num, unit = parts
    try:
        num = float(num)
    except ValueError:
        return None

    if "day" in unit:
        return num / 365
    elif "week" in unit:
        return num / 52
    elif "month" in unit:
        return num / 12
    elif "year" in unit:
        return num
    else:
        return None
    

# HANDLING COLOR AND BREED FEATURES

def learn_rare_map(series: pd.Series, threshold: float = 0.01):
    """
    Learn which categories to keep based on their relative frequency.
    Categories with frequency >= threshold are kept;
    the rest will be grouped as 'Other'.

    Parameters
    ----------
    series : pd.Series
    threshold : float, optional
        Minimum frequency threshold (default 1%)

    Returns
    -------
    dict
        Mapping information containing:
            - keep: set of frequent categories
            - coverage: share of samples covered by them
            - threshold: threshold used
            - n_unique: number of original unique categories
    """
    freq = series.value_counts(normalize=True)       # category distribution
    keep = freq[freq >= threshold].index.tolist()    # frequent categories
    coverage = float(freq.loc[keep].sum())           # total coverage
    return {
        "keep": set(keep),
        "coverage": coverage,
        "threshold": float(threshold),
        "n_unique": int(freq.size),
    }


def apply_rare_map(series: pd.Series, rare_map: dict):
    """
    Apply the learned rare-category map to a Series.

    Any value not in rare_map["keep"] will be replaced by 'Other'.
    """
    keep = rare_map["keep"]
    return series.where(series.isin(keep), other="Other")


#preprocess the data
def preprocess_shelter_data(df):
    df=df.copy()

    # --- Age ---
    df['age_years'] = df['AgeuponOutcome'].apply(convert_age_to_years)
    
# === Handling Missing Values ===

    # 'Name':
# We will have a column of ints where 1 means the animal has a name, and 0 means it does not.
    df['HasName'] = (df['Name'].fillna('') != '').astype(int)

# 'OutcomeSubtype':
# In principle, this variable could serve as an additional predictive target.
# However, due to the presence of nearly 50% missing entries, imputation is not appropriate
# and its inclusion would compromise data reliability.
# The feature is therefore excluded from further analysis to ensure consistent and interpretable results.
    if 'OutcomeSubtype' in df.columns:
      df.drop(columns=['OutcomeSubtype'], inplace=True)

# 'SexuponOutcome':
# Only one value is missing in this feature.
# To balance functionality and statistical robustness, we replace the missing entry with the most frequent category (mode imputation).
    df['SexuponOutcome'] = df['SexuponOutcome'].fillna(df['SexuponOutcome'].mode()[0])

# 'AgeuponOutcome':
# For this numerical feature, we'll apply a multivariate imputation using a K-Nearest Neighbors (KNN) approach.
# KNNImputer leverages correlated features to infer realistic values for missing ages, ensuring consistency.
# Normalization or standardization is required.

# select potential related features, then create a copy to not mess with any original data
    features_for_imputation = ['AnimalType', 'SexuponOutcome', 'age_years']
    impute_df = df[features_for_imputation].copy()

# encode catagorical features to numerical using one hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = encoder.fit_transform(impute_df[['AnimalType', 'SexuponOutcome']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['AnimalType', 'SexuponOutcome']))

# combine all into one
    combined = pd.concat([impute_df[['age_years']].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

#standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)

# apply KNN with 3 as k
    imputer = KNNImputer(n_neighbors=3)
    imputed_scaled = imputer.fit_transform(scaled)

# reverse scale
    imputed = scaler.inverse_transform(imputed_scaled)

# apply to data set
    df['age_years'] = imputed[:, 0]

# === Outlier Handling ===
# Calculate the interquartile range (IQR) for age_years
    Q1 = df['age_years'].quantile(0.25)
    Q3 = df['age_years'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

# cap extreme values instead of dropping data (removing 7.57% of data could skew data)
    df['age_years_capped'] = df['age_years'].clip(lower, upper)
# Learn the mapping and apply it on training data
    breed_map = learn_rare_map(df["Breed"], threshold=0.01)
    color_map = learn_rare_map(df["Color"], threshold=0.01)

    df["Breed_simplified"] = apply_rare_map(df["Breed"], breed_map)
    df["Color_simplified"] = apply_rare_map(df["Color"], color_map)

# HANDLING THE DATETIME FEATURE

# Parse the DateTime column into proper datetime format
    df['DateTime'] = pd.to_datetime(df['DateTime'])

# Create cyclic time features to capture temporal periodicity
#    - Hour of the day
    df['Hour'] = df['DateTime'].dt.hour
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

#    - Day of the week
    df['Weekday'] = df['DateTime'].dt.dayofweek   # Monday = 0, Sunday = 6
    df['Wday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Wday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

#    - Day of the year
    doy = df['DateTime'].dt.dayofyear
    df['DoY_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['DoY_cos'] = np.cos(2 * np.pi * doy / 365.25)

# Add a linear time feature to preserve chronological order
    t0 = df['DateTime'].min()
    df['DaysSinceStart'] = (df['DateTime'] - t0).dt.days.astype(int)

# Add useful binary flags
    df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)  # 1 = Saturday/Sunday, 0 = weekday

# Keep 'Year' to capture long-term temporal trends
    df['Year'] = df['DateTime'].dt.year

# One-hot encode all remaining categorical features
# Drop unnecessary or redundant columns before encoding
    X = pd.get_dummies(
     df.drop(columns=[
        'OutcomeType', 'AnimalID', 'Name',
        'age_years', 'Hour', 'Weekday', 'DateTime',
         'Breed', 'Color', 'AgeuponOutcome'
    ]),
    drop_first=True
)

# Target variable
    y = df['OutcomeType']

    return X, y, breed_map, color_map
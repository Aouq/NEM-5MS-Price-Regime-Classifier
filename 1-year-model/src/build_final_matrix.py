import pandas as pd
import numpy as np
import gc

# --- CONFIGURATION ---
SHIFT_STEPS = 1         # 1 step = 5 minutes look-back
TRAIN_RATIO = 0.80      # First 80% for Train, last 20% for Validation
OUTPUT_TRAIN = "Train.csv"
OUTPUT_VAL   = "Validation.csv"

print("--- MASTER DATA PIPELINE STARTED ---")

# ==========================================
# PHASE 1: ASSEMBLE MATRIX (Tiered & Smart)
# ==========================================
print("\n[1/4] Loading & Merging Data...")

df_price = pd.read_csv("VIC1_Price.csv")
df_region = pd.read_csv("VIC1_RegionSum.csv")
df_price['SETTLEMENTDATE'] = pd.to_datetime(df_price['SETTLEMENTDATE'])
df_region['SETTLEMENTDATE'] = pd.to_datetime(df_region['SETTLEMENTDATE'])

df_main = pd.merge(df_price, df_region, on=['SETTLEMENTDATE', 'REGIONID'], how='inner')

del df_price, df_region
gc.collect()

# Constraints processing
df_const = pd.read_csv("VIC1_Constraints.csv")
df_const['SETTLEMENTDATE'] = pd.to_datetime(df_const['SETTLEMENTDATE'])
df_lookup = pd.read_csv("Constraint_Lookup_Clean.csv")

lookup_map = dict(zip(df_lookup['GENCONID'], df_lookup.get('LIMITTYPE', df_lookup.get('CONSTRAINTTYPE'))))
df_const['Type'] = df_const['CONSTRAINTID'].map(lookup_map)

# Smart Rules
mask_missing = df_const['Type'].isna()
missing_ids = df_const.loc[mask_missing, 'CONSTRAINTID'].astype(str).str.upper()
df_const.loc[mask_missing & missing_ids.str.contains("RAMP"), 'Type'] = "Ramp_Rate"
df_const.loc[mask_missing & (missing_ids.str.startswith("F_") | missing_ids.str.contains("_FCAS")), 'Type'] = "FCAS"
df_const['Type'] = df_const['Type'].fillna("Other")

pivot_df = df_const.pivot_table(index='SETTLEMENTDATE', columns='Type', values='MARGINALVALUE', aggfunc='sum', fill_value=0)
pivot_df.columns = [f'Constraint_{str(c).replace(" ", "_")}' for c in pivot_df.columns]

df = pd.merge(df_main, pivot_df, left_on='SETTLEMENTDATE', right_index=True, how='left').fillna(0)
df = df.sort_values('SETTLEMENTDATE')

# ==========================================
# PHASE 2: FEATURE ENGINEERING (Cyclical)
# ==========================================
print("\n[2/4] Engineering Features...")

# 1. PRICE_STATUS Mapping
if 'PRICE_STATUS' in df.columns:
    df['PRICE_STATUS'] = df['PRICE_STATUS'].map({'FIRM': 1, 'INTERVENTION': 2}).fillna(0)

# 2. High-Resolution Cyclical Time Encoding
# Combining Hour and Minute into total minutes of the day (0-1439)
minutes_of_day = df['SETTLEMENTDATE'].dt.hour * 60 + df['SETTLEMENTDATE'].dt.minute
df['time_sin'] = np.sin(2 * np.pi * minutes_of_day / 1440.0)
df['time_cos'] = np.cos(2 * np.pi * minutes_of_day / 1440.0)

# 3. Weekday Features
df['Weekday'] = df['SETTLEMENTDATE'].dt.dayofweek
df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)

# 4. Assemble Final Matrix
df_final = pd.DataFrame()
df_final['SETTLEMENTDATE'] = df['SETTLEMENTDATE']
df_final['Target_RRP'] = df['RRP']

# Add non-lagged temporal features (Cyclical and Weekday)
time_features = ['time_sin', 'time_cos', 'Weekday', 'IsWeekend']
for t in time_features:
    df_final[t] = df[t]

# 5. Define and Shift Lagged Features
# Exclude raw time (Hour, Minute, Month), metadata (RegionID), and target (RRP)
exclude = ['SETTLEMENTDATE', 'REGIONID', 'RRP', 'Hour', 'Minute', 'Month'] + time_features
features_to_shift = [c for c in df.columns if c not in exclude]

for col in features_to_shift:
    df_final[f'{col}_Lag{SHIFT_STEPS}'] = df[col].shift(SHIFT_STEPS)

# Explicitly include Lagged Price
df_final[f'RRP_Lag{SHIFT_STEPS}'] = df['RRP'].shift(SHIFT_STEPS)

df_final = df_final.dropna()

# ==========================================
# PHASE 3: SPLIT DATA
# ==========================================
print("\n[3/4] Splitting Data...")
split_idx = int(len(df_final) * TRAIN_RATIO)
train_df = df_final.iloc[:split_idx].copy()
val_df = df_final.iloc[split_idx:].copy()

# ==========================================
# PHASE 4: SAVE (Strictly Numerical)
# ==========================================
print("\n[4/4] Saving Files...")
# Dropping SETTLEMENTDATE to ensure XGBoost receives strictly numerical inputs
train_df.drop(columns=['SETTLEMENTDATE']).to_csv(OUTPUT_TRAIN, index=False)
val_df.drop(columns=['SETTLEMENTDATE']).to_csv(OUTPUT_VAL, index=False)

print(f"SUCCESS! Created '{OUTPUT_TRAIN}' and '{OUTPUT_VAL}'.")
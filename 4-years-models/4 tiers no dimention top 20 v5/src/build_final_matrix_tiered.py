import pandas as pd
import numpy as np
import gc
import os
from path_config import ABSOLUTE_TRAIN_PATH, ABSOLUTE_VAL_PATH

# --- CONFIGURATION ---
SHIFT_STEPS = 1         # 1 step = 5 minutes look-back
TRAIN_RATIO = 0.80      # First 80% for Train, last 20% for Validation

print("--- MASTER DATA PIPELINE STARTED ---")

# ==========================================
# PHASE 1: ASSEMBLE MATRIX (Tiered & Smart)
# ==========================================
print("\n[1/4] Loading & Merging Data...")

try:
    df_price = pd.read_parquet("VIC1_Price.parquet")
    df_region = pd.read_parquet("VIC1_RegionSum.parquet")
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Data dependency missing. {e}")
    exit(1)

df_price['SETTLEMENTDATE'] = pd.to_datetime(df_price['SETTLEMENTDATE'])
df_region['SETTLEMENTDATE'] = pd.to_datetime(df_region['SETTLEMENTDATE'])

df_main = pd.merge(df_price, df_region, on=['SETTLEMENTDATE', 'REGIONID'], how='inner')

del df_price, df_region
gc.collect()

# Constraints processing
df_const = pd.read_parquet("VIC1_Constraints.parquet")
df_const['SETTLEMENTDATE'] = pd.to_datetime(df_const['SETTLEMENTDATE'])

# Flexible metadata loading
try:
    df_lookup = pd.read_parquet("Constraint_Lookup_Clean.parquet")
except FileNotFoundError:
    df_lookup = pd.read_csv("Constraint_Lookup_Clean.csv")

lookup_map = dict(zip(df_lookup['GENCONID'], df_lookup.get('LIMITTYPE', df_lookup.get('CONSTRAINTTYPE'))))
df_const['Type'] = df_const['CONSTRAINTID'].map(lookup_map)

# Smart Rules
mask_missing = df_const['Type'].isna()
missing_ids = df_const.loc[mask_missing, 'CONSTRAINTID'].astype(str).str.upper()
df_const.loc[mask_missing & missing_ids.str.contains("RAMP"), 'Type'] = "Ramp_Rate"
df_const.loc[mask_missing & (missing_ids.str.startswith("F_") | missing_ids.str.contains("_FCAS")), 'Type'] = "FCAS"
df_const['Type'] = df_const['Type'].fillna("Other")

# --- EXPLICIT CAUSAL EXTRACTION (TOP 20) ---
# 1. Define Temporal Boundary to prevent Look-Ahead Bias
unique_dates = df_main['SETTLEMENTDATE'].sort_values().unique()
split_index = int(len(unique_dates) * TRAIN_RATIO)
t_split = unique_dates[split_index]

# 2. Isolate Historical Super-Spikes in the Training Data
train_main = df_main[df_main['SETTLEMENTDATE'] < t_split]
# Assuming Super-Spike is defined by extreme prices (e.g., RRP >= 300)
historical_spike_dates = train_main[train_main['RRP'] >= 300.0]['SETTLEMENTDATE']

# 3. Identify the Worst Offenders during those specific spike intervals
spike_const_df = df_const[df_const['SETTLEMENTDATE'].isin(historical_spike_dates)]
top_20_series = spike_const_df.groupby('CONSTRAINTID')['MARGINALVALUE'].apply(lambda x: x.abs().sum()).nlargest(20)
top_20_ids = top_20_series.index.tolist()

print(f"\n[HYBRID ARCHITECTURE] Explicitly extracted Top 20 Historical Constraints:")
for i, cid in enumerate(top_20_ids):
    print(f"  {i+1}. {cid}")

# 4. Apply the Top 20 Overrides
# If a constraint is in the Top 20, its Type becomes its literal ID.
df_const['Type'] = np.where(df_const['CONSTRAINTID'].isin(top_20_ids), 
                            df_const['CONSTRAINTID'], 
                            df_const['Type'])

# --- 4-TIER STATISTICAL MAGNITUDE CLASSIFICATION (For 'Other' Only) ---
mask_other = df_const['Type'] == 'Other'

if mask_other.any():
    mv_global = df_const.loc[mask_other, 'MARGINALVALUE'].abs()
    binding_mv = mv_global[mv_global > 0]
    
    if not binding_mv.empty:
        q99 = np.percentile(binding_mv, 99)
        q90 = np.percentile(binding_mv, 90)
        q50 = np.percentile(binding_mv, 50)
        
        mv_abs = df_const['MARGINALVALUE'].abs()
        
        df_const.loc[mask_other & (mv_abs >= q99), 'Type'] = "Other_Super_High"
        df_const.loc[mask_other & (mv_abs >= q90) & (mv_abs < q99), 'Type'] = "Other_High"
        df_const.loc[mask_other & (mv_abs >= q50) & (mv_abs < q90), 'Type'] = "Other_Med"
        df_const.loc[mask_other & (mv_abs < q50), 'Type'] = "Other_Low"

# Pivot Transformation
# By avoiding .abs() here, the Top 20 explicit columns will retain their directional vectors (positive/negative signs).
pivot_df = df_const.pivot_table(index='SETTLEMENTDATE', columns='Type', values='MARGINALVALUE', aggfunc='sum', fill_value=0)
pivot_df.columns = [f'Constraint_{str(c).replace(" ", "_")}' for c in pivot_df.columns]

df = pd.merge(df_main, pivot_df, left_on='SETTLEMENTDATE', right_index=True, how='left').fillna(0)
df = df.sort_values('SETTLEMENTDATE')

# ==========================================
# PHASE 2: FEATURE ENGINEERING (Cyclical)
# ==========================================
print("\n[2/4] Engineering Features...")

if 'PRICE_STATUS' in df.columns:
    df['PRICE_STATUS'] = df['PRICE_STATUS'].map({'FIRM': 1, 'INTERVENTION': 2}).fillna(0)

minutes_of_day = df['SETTLEMENTDATE'].dt.hour * 60 + df['SETTLEMENTDATE'].dt.minute
df['time_sin'] = np.sin(2 * np.pi * minutes_of_day / 1440.0)
df['time_cos'] = np.cos(2 * np.pi * minutes_of_day / 1440.0)

df['Weekday'] = df['SETTLEMENTDATE'].dt.dayofweek
df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)

df_final = pd.DataFrame()
df_final['SETTLEMENTDATE'] = df['SETTLEMENTDATE']
df_final['Target_RRP'] = df['RRP']

time_features = ['time_sin', 'time_cos', 'Weekday', 'IsWeekend']
for t in time_features:
    df_final[t] = df[t]

exclude = ['SETTLEMENTDATE', 'REGIONID', 'RRP', 'Hour', 'Minute', 'Month'] + time_features
features_to_shift = [c for c in df.columns if c not in exclude]

for col in features_to_shift:
    df_final[f'{col}_Lag{SHIFT_STEPS}'] = df[col].shift(SHIFT_STEPS)

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
train_df.drop(columns=['SETTLEMENTDATE']).to_parquet(ABSOLUTE_TRAIN_PATH, index=False)
val_df.drop(columns=['SETTLEMENTDATE']).to_parquet(ABSOLUTE_VAL_PATH, index=False)

print(f"SUCCESS! Created Training and Validation matrices at configured destination.")
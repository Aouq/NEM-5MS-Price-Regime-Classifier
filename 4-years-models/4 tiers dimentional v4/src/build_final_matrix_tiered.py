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

# --- DUAL-VECTOR MAPPING (Base Type + Inequality Operator) ---
# 1. Extract Base Limit Type
limit_map = dict(zip(df_lookup['GENCONID'], df_lookup.get('LIMITTYPE', df_lookup.get('CONSTRAINTTYPE'))))
df_const['BaseType'] = df_const['CONSTRAINTID'].map(limit_map)

# 2. Extract Inequality Operator
operator_map = dict(zip(df_lookup['GENCONID'], df_lookup.get('CONSTRAINTTYPE', 'UNK')))
df_const['RawOperator'] = df_const['CONSTRAINTID'].map(operator_map).fillna('UNK')

# 3. Lexical Translation (Format symbols into safe column suffixes)
operator_clean = {
    '<=': '_LE',
    '>=': '_GE',
    '=>': '_GE', # Lexical catch for topographical variations
    '=': '_EQ',
    'UNK': '_UNK'
}
df_const['Op_Suffix'] = df_const['RawOperator'].map(operator_clean).fillna('_UNK')

# --- SMART RULES ---
mask_missing = df_const['BaseType'].isna()
missing_ids = df_const.loc[mask_missing, 'CONSTRAINTID'].astype(str).str.upper()
df_const.loc[mask_missing & missing_ids.str.contains("RAMP"), 'BaseType'] = "Ramp_Rate"
df_const.loc[mask_missing & (missing_ids.str.startswith("F_") | missing_ids.str.contains("_FCAS")), 'BaseType'] = "FCAS"

# Base Classification
df_const['BaseType'] = df_const['BaseType'].fillna("Other")

# --- 4-TIER STATISTICAL MAGNITUDE CLASSIFICATION ---
mask_other = df_const['BaseType'] == 'Other'

if mask_other.any():
    mv_global = df_const.loc[mask_other, 'MARGINALVALUE'].abs()
    binding_mv = mv_global[mv_global > 0]
    
    if not binding_mv.empty:
        q99 = np.percentile(binding_mv, 99)
        q90 = np.percentile(binding_mv, 90)
        q50 = np.percentile(binding_mv, 50)
        
        mv_abs = df_const['MARGINALVALUE'].abs()
        
        # We classify the BaseType, preserving the Op_Suffix for later concatenation
        df_const.loc[mask_other & (mv_abs >= q99), 'BaseType'] = "Other_Super_High"
        df_const.loc[mask_other & (mv_abs >= q90) & (mv_abs < q99), 'BaseType'] = "Other_High"
        df_const.loc[mask_other & (mv_abs >= q50) & (mv_abs < q90), 'BaseType'] = "Other_Med"
        df_const.loc[mask_other & (mv_abs < q50), 'BaseType'] = "Other_Low"

# --- FEATURE COMBINATION ---
# Concatenate the physical parameter with its mathematical boundary
df_const['Type'] = df_const['BaseType'].astype(str) + df_const['Op_Suffix']

# Pivot Transformation
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
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# --- CONFIGURATION (Aggressive Comparison) ---
TRAIN_FILE = "Train.csv"
VAL_FILE   = "Validation.csv"
# Unique output names for comparison
MODEL_OUTPUT = "xgboost_regime_model_aggressive.json"
PLOT_LEARNING = "Learning_Curve_aggressive.png"
PLOT_MATRIX = "Confusion_Matrix_aggressive.png"

HIGH_THRESHOLD  = 117.0
SPIKE_THRESHOLD = 300.0

print("--- TRAINING AGGRESSIVE XGBOOST: SPIKE PERSISTENCE FOCUS ---")

# 1. Data Ingestion
df_train = pd.read_csv(TRAIN_FILE)
df_val   = pd.read_csv(VAL_FILE)

def get_regime(price):
    if price >= SPIKE_THRESHOLD: return 2 
    if price >= HIGH_THRESHOLD:  return 1 
    return 0

y_train = df_train['Target_RRP'].apply(get_regime)
y_val   = df_val['Target_RRP'].apply(get_regime)
X_train = df_train.drop(columns=['Target_RRP'])
X_val   = df_val.drop(columns=['Target_RRP'])

# 2. Aggressive Class Weighting
# Start with balanced weights, then apply a multiplier to the Spike class (Class 2)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
spike_multiplier = 5.0  # Force the model to prioritize spike recall over precision
sample_weights[y_train == 2] *= spike_multiplier

# 3. Model Architecture (Aggressive Hyperparameters)
model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=12,               # Increased depth to capture intricate constraint triggers
    objective='multi:softprob',
    num_class=3,
    device="cuda",              # NVIDIA GPU acceleration
    tree_method="hist", 
    early_stopping_rounds=75,    # Allow more iterations for fine-tuning
    random_state=42,
    n_jobs=-1
)

# 4. Training
evals_result = {}
model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# 5. Diagnostic Plot: Learning Curve
results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
plt.figure(figsize=(10, 6))
plt.plot(range(0, epochs), results['validation_0']['mlogloss'], label='Aggressive Train')
plt.plot(range(0, epochs), results['validation_1']['mlogloss'], label='Aggressive Validation')
plt.ylabel('LogLoss')
plt.title('Convergence: Aggressive Model (Spike Multiplier = 5x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(PLOT_LEARNING)

# 6. Model Persistence
model.save_model(MODEL_OUTPUT)
print(f"   -> Aggressive model saved to '{MODEL_OUTPUT}'")

# 7. Comparison Metrics
y_pred = model.predict(X_val)
print("\n--- AGGRESSIVE PERFORMANCE REPORT ---")
print(classification_report(y_val, y_pred, target_names=['Normal', 'High', 'Spike']))

# Confusion Matrix for Comparison
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Normal', 'High', 'Spike'],
            yticklabels=['Normal', 'High', 'Spike'])
plt.title('Regime Transition Matrix (Aggressive Weighting)')
plt.savefig(PLOT_MATRIX)
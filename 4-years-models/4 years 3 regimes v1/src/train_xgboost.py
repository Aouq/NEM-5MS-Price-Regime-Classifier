import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# --- CONFIGURATION (Formal Parameters) ---
TRAIN_FILE = "Train.csv"
VAL_FILE   = "Validation.csv"
MODEL_OUTPUT = "xgboost_regime_model.json"

# Thresholds calibrated from price distribution analysis
HIGH_THRESHOLD  = 117.0
SPIKE_THRESHOLD = 300.0

print("--- TRAINING XGBOOST: REGIME PERSISTENCE ENGINE ---")

# 1. Data Ingestion
df_train = pd.read_csv(TRAIN_FILE)
df_val   = pd.read_csv(VAL_FILE)

def get_regime(price):
    """Categorizes RRP into discrete market regimes."""
    if price >= SPIKE_THRESHOLD: return 2 # Spike Persistence
    if price >= HIGH_THRESHOLD:  return 1 # High-Price Regime
    return 0                              # Normal Operations

y_train = df_train['Target_RRP'].apply(get_regime)
y_val   = df_val['Target_RRP'].apply(get_regime)

# Feature Matrix Isolation (Target_RRP is excluded to prevent data leakage)
X_train = df_train.drop(columns=['Target_RRP'])
X_val   = df_val.drop(columns=['Target_RRP'])

# 2. Stochastic Weighting for Class Imbalance
# Balances the influence of rare Spike events (0.98% frequency)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 3. Model Architecture (CUDA Acceleration)
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,                # Increased depth for complex regime drivers
    objective='multi:softprob',
    num_class=3,
    # NVIDIA CUDA Configuration
    device="cuda",
    tree_method="hist", 
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

# 4. Training and Evaluation Monitoring
evals_result = {}  # Container for learning curve metrics

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# 5. Diagnostic Visualization: Learning Curve
results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Training Loss')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Validation Loss')
plt.ylabel('Multi-class Logarithmic Loss (LogLoss)')
plt.xlabel('Iterations (Gradient Steps)')
plt.title('XGBoost Convergence Diagnostics')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Learning_Curve.png')
print("   -> Learning curve generated: 'Learning_Curve.png'")

# 6. Model Persistence
model.save_model(MODEL_OUTPUT)
print(f"   -> Model weight matrix saved to '{MODEL_OUTPUT}'")

# 7. Final Performance Metrics
y_pred = model.predict(X_val)
print("\n--- CLASSIFICATION PERFORMANCE REPORT ---")
print(f"Global Accuracy: {accuracy_score(y_val, y_pred):.2%}")
print(classification_report(y_val, y_pred, target_names=['Normal', 'High', 'Spike']))

# Confusion Matrix for Regime Transitions
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'High', 'Spike'],
            yticklabels=['Normal', 'High', 'Spike'])
plt.ylabel('Actual Regime')
plt.xlabel('Predicted Regime')
plt.title('Regime Transition Matrix')
plt.savefig('Confusion_Matrix.png')
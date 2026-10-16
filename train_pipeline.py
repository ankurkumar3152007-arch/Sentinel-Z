import pandas as pd
import numpy as np
import lightgbm as lgb
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import time
import warnings
warnings.filterwarnings('ignore')

print("=== SENTINEL-Z: REAL-WORLD DATA INGESTION ===")

# 1. Load and Process Real PaySim Data
try:
    print("\n1. Loading paysim.csv (This may take 10-20 seconds)...")
    df = pd.read_csv('paysim.csv')
except FileNotFoundError:
    print("❌ ERROR: paysim.csv not found! Please download it from Kaggle and place it in this folder.")
    exit()

# Fraud in PaySim only occurs in TRANSFER and CASH_OUT types. 
# We filter the dataset to optimize memory and training speed.
df = df.loc[(df['type'] == 'TRANSFER') | (df['type'] == 'CASH_OUT')]

print(f"Data loaded: {len(df)} relevant transactions found.")
print("\n2. Engineering behavioral features...")

# Feature Engineering: Track ledger discrepancies caused by rapid money movement
df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

features = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 
    'errorBalanceOrig', 'errorBalanceDest'
]

X = df[features].astype(np.float32)
y = df['isFraud']

# 3. Train the Classifier
print("\n3. Training LightGBM Classifier on real fraud patterns...")
model = lgb.LGBMClassifier(
    n_estimators=150, 
    max_depth=5, 
    learning_rate=0.1, 
    scale_pos_weight=10, # Heavily penalize the model for missing actual fraud
    verbose=-1
)
model.fit(X, y)
print("Model training complete.")

# 4. Compile to ONNX
print("\n4. Compiling model to ONNX runtime format...")
initial_types = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=12)

with open("sentinel_z_core.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("Model saved as 'sentinel_z_core.onnx'.")

# 5. Live Inference Test (Testing a known fraud signature)
print("\n5. Running ONNX latency benchmark...")
sess = ort.InferenceSession("sentinel_z_core.onnx")
input_name = sess.get_inputs()[0].name

# Simulating a classic money-mule attack:
# High amount (500k), completely drains original account, destination account ledger mismatch
test_tx = np.array([[500000.0, 500000.0, 0.0, 0.0, 0.0, 0.0, 500000.0]], dtype=np.float32)

_ = sess.run(None, {input_name: test_tx}) # Warmup

start_time = time.perf_counter()
pred_onnx = sess.run(None, {input_name: test_tx})
end_time = time.perf_counter()

latency_ms = (end_time - start_time) * 1000
fraud_score_dict = pred_onnx[1][0]
fraud_probability = fraud_score_dict[1] * 100 

print(f"\n--- SENTINEL-Z INFERENCE RESULTS ---")
print(f"Risk Score:      {fraud_probability:.2f}% (High confidence expected)")
print(f"Execution Time:  {latency_ms:.4f} ms")

if latency_ms < 3.0:
    print("System Status:   ✅ SLA MET")
else:
    print("System Status:   ❌ SLA FAILED")
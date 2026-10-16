from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
import time

app = FastAPI(title="Sentinel-Z In-Transaction Interceptor")

# Load ONNX session once at startup to keep memory warm
session = ort.InferenceSession("sentinel_z_core.onnx")
input_name = session.get_inputs()[0].name

class TransactionPayload(BaseModel):
    transaction_id: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.post("/v1/transaction/intercept")
async def intercept_transaction(tx: TransactionPayload):
    start_time = time.perf_counter()

    # 1. In-memory feature calculation
    error_orig = tx.newbalanceOrig + tx.amount - tx.oldbalanceOrg
    error_dest = tx.oldbalanceDest + tx.amount - tx.newbalanceDest

    features = np.array([[
        tx.amount,
        tx.oldbalanceOrg,
        tx.newbalanceOrig,
        tx.oldbalanceDest,
        tx.newbalanceDest,
        error_orig,
        error_dest
    ]], dtype=np.float32)

    # 2. Sub-millisecond ONNX inference
    pred_onnx = session.run(None, {input_name: features})
    
    # Extract fraud probability (class 1)
    fraud_prob = float(pred_onnx[1][0][1]) * 100.0

    # 3. Deterministic decision gating
    if fraud_prob < 30.0:
        action = "ALLOW"
        routing = "GREEN_PATH"
    elif fraud_prob <= 75.0:
        action = "STEP_UP_MFA"
        routing = "YELLOW_PATH"
    else:
        action = "BLOCK_MULE"
        routing = "RED_PATH"

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "transaction_id": tx.transaction_id,
        "fraud_risk_score": round(fraud_prob, 2),
        "decision": action,
        "routing": routing,
        "inference_latency_ms": round(latency_ms, 4),
        "sla_met": latency_ms < 3.0
    }
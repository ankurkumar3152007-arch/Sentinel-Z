import platform
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    import s390x_ort as ort

print(f"=== SENTINEL-Z BENCHMARK ({platform.machine()}) ===")
sess = ort.InferenceSession("sentinel_z_core.onnx")
input_name = sess.get_inputs()[0].name

# Test payload: large transaction draining account
test_tx = np.array([[500000.0, 500000.0, 0.0, 0.0, 0.0, 0.0, 500000.0]], dtype=np.float32)

# Warmup runs
for _ in range(10):
    _ = sess.run(None, {input_name: test_tx})

latencies = []
for _ in range(1000):
    t0 = time.perf_counter()
    pred = sess.run(None, {input_name: test_tx})
    latencies.append((time.perf_counter() - t0) * 1000.0)

avg_ms = np.mean(latencies)
p99_ms = np.percentile(latencies, 99)
risk_score = float(pred[1][0][1]) * 100.0

print(f"Platform:         {platform.system()} {platform.machine()}")
print(f"Fraud Risk Score: {risk_score:.2f}%")
print(f"Avg Latency:      {avg_ms:.4f} ms")
print(f"P99 Tail Latency: {p99_ms:.4f} ms")
print(f"SLA Status:       {'PASS (<3ms)' if p99_ms < 3.0 else 'FAIL'}")

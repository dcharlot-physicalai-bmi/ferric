import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np

# Model A: two-input matmul (512,512)x(512,512) fp16 — is a dynamic-B matmul ANE-eligible?
@mb.program(input_specs=[mb.TensorSpec(shape=(512, 512)), mb.TensorSpec(shape=(512, 512))])
def matmul_prog(x, y):
    return mb.matmul(x=x, y=y, name="out")

m = ct.convert(matmul_prog, convert_to="mlprogram",
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.macOS13)
m.save("matmul512.mlpackage")
print("matmul512 saved")

# Model B: fixed-weight linear+relu x[512,512]·Wt+b, weights baked — the classic ANE pattern
rng = np.random.default_rng(7)
W = (rng.standard_normal((512, 512)) * 0.05).astype(np.float32)

@mb.program(input_specs=[mb.TensorSpec(shape=(512, 512))])
def linear_prog(x):
    h = mb.matmul(x=x, y=W)
    return mb.relu(x=h, name="out")

m2 = ct.convert(linear_prog, convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS13)
m2.save("linear512.mlpackage")
np.save("linear_w.npy", W)
print("linear512 saved")

import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np

rng = np.random.default_rng(7)

# conv 3x3, 64ch, 56x56 — the classic ANE workload
W = (rng.standard_normal((64, 64, 3, 3)) * 0.05).astype(np.float32)
@mb.program(input_specs=[mb.TensorSpec(shape=(1, 64, 56, 56))])
def conv3(x):
    return mb.conv(x=x, weight=W, strides=[1, 1], pad_type="same", name="out")
ct.convert(conv3, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
           minimum_deployment_target=ct.target.macOS13).save("conv3x3.mlpackage")

# linear as 1x1 conv over a spatial map
W1 = (rng.standard_normal((512, 512, 1, 1)) * 0.05).astype(np.float32)
@mb.program(input_specs=[mb.TensorSpec(shape=(1, 512, 32, 32))])
def conv1(x):
    return mb.conv(x=x, weight=W1, strides=[1, 1], pad_type="valid", name="out")
ct.convert(conv1, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
           minimum_deployment_target=ct.target.macOS13).save("conv1x1.mlpackage")

# attention-shaped rank-4 batched matmul, both inputs dynamic
@mb.program(input_specs=[mb.TensorSpec(shape=(1, 8, 512, 64)), mb.TensorSpec(shape=(1, 8, 64, 512))])
def mm4(x, y):
    return mb.matmul(x=x, y=y, name="out")
ct.convert(mm4, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
           minimum_deployment_target=ct.target.macOS13).save("matmul4d.mlpackage")
print("done")

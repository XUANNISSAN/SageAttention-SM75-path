from .core import sageattn, sageattn_varlen
from .core import sageattn_qk_int8_pv_fp16_triton
from .core import sageattn_qk_int8_pv_fp16_cuda 
from .core import sageattn_qk_int8_pv_fp8_cuda
from .core import sageattn_qk_int8_pv_fp8_cuda_sm90
# Add SM75 if built
try:
    from .core import sageattn_qk_int8_pv_fp16_cuda_sm75
except ImportError:
    pass # If SM75 kernel wasn't built, don't expose it

__version__ = "2.1.1" # Or update if making a new release

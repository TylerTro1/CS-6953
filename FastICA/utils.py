# utils.py
from __future__ import annotations
import numpy as np

def validate_input(var_index, min_val, max_val, var_str):
    if var_index != int(var_index) or var_index <= 0:
        raise ValueError(f"Input variable {var_str} must be a positive integer")

    var_index = int(var_index)

    if var_index < min_val:
        raise ValueError(f"Input variable {var_str} must be greater than {min_val}")

    if var_index > max_val:
        raise ValueError(f"Input variable {var_str} must be less than {max_val}")

def mat_struct_to_dict(obj):
    """
    Convert scipy.io.loadmat MATLAB structs into nested python dicts.
    Handles numpy structured arrays + mat_struct-like objects.
    """
    # Already a dict
    if isinstance(obj, dict):
        return {k: mat_struct_to_dict(v) for k, v in obj.items()}

    # numpy arrays: squeeze singletons
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # Often loadmat gives object arrays of structs
            if obj.size == 1:
                return mat_struct_to_dict(obj.item())
            return [mat_struct_to_dict(x) for x in obj.flat]
        # structured array
        if obj.dtype.names is not None:
            if obj.size == 1:
                obj = obj.item()
            return {name: mat_struct_to_dict(obj[name]) for name in obj.dtype.names}
        return obj

    # mat_struct-like objects (have _fieldnames)
    if hasattr(obj, "_fieldnames"):
        return {name: mat_struct_to_dict(getattr(obj, name)) for name in obj._fieldnames}

    return obj

def gray_code(n: np.ndarray) -> np.ndarray:
    return n ^ (n >> 1)

def int_to_bits_lsb_first(x_int: np.ndarray, bits_per_sym: int) -> np.ndarray:
    """
    MATLAB de2bi default is often LSB-first. This returns shape (bits_per_sym, N)
    matching MATLAB de2bi(...).' convention.
    """
    x_int = np.asarray(x_int, dtype=np.int64).ravel()
    bits = ((x_int[:, None] >> np.arange(bits_per_sym)) & 1).astype(np.uint8)  # (N,b)
    return bits.T  # (b,N)

def psk_gray_demod(z: np.ndarray, M: int) -> np.ndarray:
    """
    Approximate MATLAB pskdemod(z, M, 0, 'gray').
    Returns Gray-coded integers in [0,M-1].
    """
    z = np.asarray(z, dtype=np.complex128)
    ang = np.angle(z) % (2*np.pi)
    k = np.floor((ang / (2*np.pi)) * M + 0.5).astype(int) % M  # nearest symbol index
    return gray_code(k)

def conv_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    MATLAB conv(x,h,'same') equivalent.
    """
    from scipy.signal import convolve
    return convolve(x, h, mode="same")

def downsample_symbols(y: np.ndarray, sps: int) -> np.ndarray:
    """
    MATLAB Y(:, sps:sps:end) (1-based indexing) => Python start at sps-1.
    Works with 1D or 2D arrays; downsamples last axis.
    """
    return y[..., (sps - 1)::sps]
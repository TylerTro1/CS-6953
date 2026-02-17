# separator.py
from __future__ import annotations
import os
import numpy as np

def per_frame_separator(input_filename: str, Nr: int,
                        read_data_fn,
                        complex_fastica_fn,
                        n_sources: int = 2) -> np.ndarray:
    # Read IQ data (Y: Nr x Nsamples)
    Y = read_data_fn(input_filename, Nr, "iqdata")
    Y = np.asarray(Y, dtype=np.complex128)

    # Run ICA
    sig_hat = complex_fastica_fn(Y, n_sources)
    sig_hat = np.asarray(sig_hat, dtype=np.complex128)

    # Normalize each separated component to unit average power
    for sI in range(sig_hat.shape[0]):
        power = np.mean(np.abs(sig_hat[sI, :])**2)
        if power > 0:
            sig_hat[sI, :] = sig_hat[sI, :] / np.sqrt(power)

    return sig_hat

def sig_separator(
    input_directory: str,
    output_directory: str,
    separation_opt: str,
    alpha_index: int,
    frame_len: int,
    set_index: int,
    n_frames: int,
    nR: int,
    per_frame_separator_fn,
    write_data_fn,
):
    suffix_str = ["A", "B"]
    os.makedirs(output_directory, exist_ok=True)

    Y0_last = None

    for ff in range(1, n_frames + 1):
        input_filename = os.path.join(
            input_directory,
            f"input_frameLen_{frame_len}_setIndex_{set_index}_alphaIndex_{alpha_index}_frame{ff}"
        )

        Y0 = per_frame_separator_fn(input_filename, nR)
        Y0_last = Y0

        for oo in range(Y0.shape[0]):
            curr_suffix = suffix_str[oo] if oo < len(suffix_str) else str(oo)

            input_struct = {
                "sample_rate": 25e6,
                "description": "separatedSignals",
                "Nr": 1,
            }

            output_filename = os.path.join(
                output_directory,
                f"output{curr_suffix}_frameLen_{frame_len}_setIndex_{set_index}_alphaIndex_{alpha_index}_frame{ff}"
            )

            sig_row = Y0[oo, :][None, :]  # (1, Nsamples)
            write_data_fn(output_filename, sig_row, input_struct)

    return Y0_last
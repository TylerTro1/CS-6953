# evaluation.py
from __future__ import annotations
import os
import numpy as np
import galois

from utils import (
    validate_input, mat_struct_to_dict,
    conv_same, downsample_symbols,
    psk_gray_demod, int_to_bits_lsb_first
)

def evaluate_separation(output_directory: str,
                        alpha_index: int,
                        frame_len: int,
                        set_index: int,
                        n_sources: int,
                        params,
                        read_data_fn,
                        gRRC: np.ndarray,
                        plot_flag: bool = False):
    """
    Port of evaluateSeparation.m

    NOTE: Exact BCH behavior depends on matching MATLAB's BCH code definition.
    This version uses galois.BCH(n,k). If results are off, we may need to
    match generator polynomial exactly.
    """
    validate_input(alpha_index, 1, 25, "alphaIndex")
    validate_input(frame_len, 1, 128, "frameLen")
    validate_input(set_index, 1, 20, "setIndex")

    params = mat_struct_to_dict(params)

    nwords = int(frame_len)
    M = int(np.squeeze(params["M"]))
    n = int(np.squeeze(params["n"]))
    k = int(np.squeeze(params["k"]))
    t = int(np.squeeze(params["t"]))  # informational; galois infers from (n,k) construction
    n_preamble_bits = int(np.squeeze(params["nPreambleBits"]))
    sps = int(np.squeeze(params["sps"]))
    noise_pwr = float(np.squeeze(params["noisePwr"]))
    n_frames = int(np.squeeze(params["nFrames"]))

    tx_syms_preamble = np.asarray(params["txSymsPreamble"]).ravel()

    bits_per_sym = int(np.log2(M))
    n_syms_preamble = int(n_preamble_bits // bits_per_sym)

    suffix = ["A", "B"]

    GF2 = galois.GF(2)
    bch = galois.BCH(n, k)

    error_count = np.zeros(n_frames, dtype=int)

    true_bits = np.asarray(params["trueBits"])
    # expected MATLAB indexing: trueBits(:, :, ff) => shape (k, nwords, nFrames)
    # We'll assume that. If yours differs, adjust here.
    if true_bits.ndim != 3:
        raise ValueError(f"params['trueBits'] expected 3D, got shape {true_bits.shape}")

    for ff in range(1, n_frames + 1):
        num_errors = np.zeros(n_sources, dtype=int)

        for cc in range(1, n_sources + 1):
            curr_suffix = suffix[cc - 1]

            base = (f"output{curr_suffix}_frameLen_{frame_len}_setIndex_{set_index}"
                    f"_alphaIndex_{alpha_index}_frame{ff}")
            separated_filename = os.path.join(output_directory, base)

            Y0 = read_data_fn(separated_filename, 1, "iqdata")
            y0 = np.asarray(Y0[0, :], dtype=np.complex128)

            # matched filter
            y = conv_same(y0, gRRC)

            # downsample
            rx_syms = downsample_symbols(y[None, :], sps)[0, :]

            # preamble
            rx_pre = rx_syms[:n_syms_preamble]
            hratio = rx_pre / tx_syms_preamble[:n_syms_preamble]
            Hest = np.mean(hratio)  # scalar (nOutputs=1 in reference)

            # payload
            rx_payload0 = rx_syms[n_syms_preamble:]

            # scalar MMSE equalizer
            denom = (Hest * np.conj(Hest) + noise_pwr)
            rx_payload = np.conj(Hest) * (rx_payload0 / denom)

            # normalize payload power
            pwr = np.mean(np.abs(rx_payload)**2)
            if pwr > 0:
                rx_payload = rx_payload / np.sqrt(pwr)

            # demod
            rx_int = psk_gray_demod(rx_payload, M)

            # int -> bits (bits_per_sym, Nsyms)
            rx_bits = int_to_bits_lsb_first(rx_int, bits_per_sym)

            # vectorize column-wise
            rx_bits_vect = rx_bits.reshape(-1, order="F")

            # reshape into codewords (nwords, n)
            needed = nwords * n
            rx_bits_vect = rx_bits_vect[:needed]
            codewords = rx_bits_vect.reshape((nwords, n), order="F")
            codewords_gf = GF2(codewords)

            decoded = bch.decode(codewords_gf)  # (nwords, k)

            tx = true_bits[:, :, ff - 1]        # (k, nwords)
            tx_gf = GF2(tx.T)                   # (nwords, k)

            num_errors[cc - 1] = int(np.sum(decoded != tx_gf))

        error_count[ff - 1] = int(np.min(num_errors))
        print(f"Frame {ff}: min bit errors = {error_count[ff-1]}")

    ber = float(np.mean(error_count) / (nwords * k))
    frame_success_rate = float(np.mean(error_count == 0))

    print(f"Bit Error Rate is {ber}")
    print(f"Frame Success Rate is {frame_success_rate}")
    return frame_success_rate, ber
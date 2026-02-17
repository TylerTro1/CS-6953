# main.py
from __future__ import annotations

import os
import argparse
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from rf_io import gRRC, read_data, write_data
from fastICA import complex_fastica
from separator import per_frame_separator, sig_separator
from eval import evaluate_separation
from utils import mat_struct_to_dict


def main():
    ap = argparse.ArgumentParser()

    # Make defaults relative to this file (robust on CHPC)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Defaults tailored to your current layout:
    # CS-6953/
    #   frameLen_128/setIndex_1/...
    #   soiParamFiles/soiParams128.mat (Octave ASCII)
    ap.add_argument("--input_dir", default=os.path.join(BASE_DIR, "frameLen_128", "setIndex_1"))
    ap.add_argument("--output_dir", default=os.path.join(BASE_DIR, "frameLen_128", "setIndex_1", "sepOutput"))
    ap.add_argument("--soi_dir", default=os.path.join(BASE_DIR, "soiParamFiles"))

    ap.add_argument("--frame_lens", nargs="+", type=int, default=[128])
    ap.add_argument("--set_list", nargs="+", type=int, default=[1])
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--nR", type=int, default=4)
    ap.add_argument("--n_sources", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123456789)
    ap.add_argument("--alpha_start", type=int, default=1)
    ap.add_argument("--alpha_end", type=int, default=25)

    # IMPORTANT: when set, we will NOT load soiParams/stats (since they are Octave ASCII .mat)
    ap.add_argument("--no_eval", action="store_true")

    args = ap.parse_args()

    # Deterministic RNG (matches MATLAB seeding intent)
    rng = np.random.default_rng(args.seed)

    def complex_fastica_det(xin, n):
        return complex_fastica(xin, n, rng=rng)

    def per_frame_sep_det(input_filename, Nr):
        return per_frame_separator(
            input_filename=input_filename,
            Nr=Nr,
            read_data_fn=read_data,
            complex_fastica_fn=complex_fastica_det,
            n_sources=args.n_sources
        )

    # If we're only separating, do NOT allocate evaluation arrays or load .mat files
    if args.no_eval:
        for frame_len in args.frame_lens:
            for alpha_index in range(args.alpha_start, args.alpha_end + 1):
                for set_index in args.set_list:
                    sig_separator(
                        input_directory=args.input_dir,
                        output_directory=args.output_dir,
                        separation_opt="ICA",
                        alpha_index=alpha_index,
                        frame_len=frame_len,
                        set_index=set_index,
                        n_frames=args.n_frames,
                        nR=args.nR,
                        per_frame_separator_fn=per_frame_sep_det,
                        write_data_fn=write_data
                    )
        print("Finished separation-only run (--no_eval).")
        return

    # ---- Evaluation mode (will fail with Octave ASCII .mat unless you convert or implement a loader)
    frame_success_rate_M = np.zeros((len(args.frame_lens), args.alpha_end, len(args.set_list)))
    ber_M = np.zeros((len(args.frame_lens), args.alpha_end, len(args.set_list)))
    sinr_thresh = np.full((len(args.frame_lens),), np.nan, dtype=float)

    for ww, frame_len in enumerate(args.frame_lens):
        soi_path = os.path.join(args.soi_dir, f"soiParams{frame_len}.mat")
        stats_path = os.path.join(args.soi_dir, f"stats{frame_len}.mat")

        # NOTE: scipy.io.loadmat cannot read Octave ASCII struct/cell .mat files.
        # If you converted them to real MAT files, update filenames here (e.g. soiParams128_v5.mat).
        soi_mat = loadmat(soi_path, struct_as_record=False, squeeze_me=True)
        stats_mat = loadmat(stats_path, struct_as_record=False, squeeze_me=True)

        soi_params_part = soi_mat["soiParamsPart"]
        median_sinr = stats_mat["medianSINR"]

        for alpha_index in range(args.alpha_start, args.alpha_end + 1):
            for ii, set_index in enumerate(args.set_list):
                # 1) run separator -> writes outputA/outputB files
                sig_separator(
                    input_directory=args.input_dir,
                    output_directory=args.output_dir,
                    separation_opt="ICA",
                    alpha_index=alpha_index,
                    frame_len=frame_len,
                    set_index=set_index,
                    n_frames=args.n_frames,
                    nR=args.nR,
                    per_frame_separator_fn=per_frame_sep_det,
                    write_data_fn=write_data
                )

                # 2) get current params struct from soiParamsPart(alpha_index, set_index)
                curr_params = soi_params_part[alpha_index - 1, set_index - 1]
                curr_params = mat_struct_to_dict(curr_params)

                # 3) evaluate
                fsr, ber = evaluate_separation(
                    output_directory=args.output_dir,
                    alpha_index=alpha_index,
                    frame_len=frame_len,
                    set_index=set_index,
                    n_sources=args.n_sources,
                    params=curr_params,
                    read_data_fn=read_data,
                    gRRC=gRRC,
                    plot_flag=False
                )

                frame_success_rate_M[ww, alpha_index - 1, ii] = fsr
                ber_M[ww, alpha_index - 1, ii] = ber

        mean_frame_success_rate = frame_success_rate_M[ww, :, :].mean(axis=1)

        # MATLAB median(medianSINR,2) => median across columns per row
        median_sinr_all = np.median(median_sinr, axis=1).squeeze()

        plt.figure()
        plt.plot(median_sinr_all, mean_frame_success_rate)
        plt.grid(True)
        plt.xlabel("Median SINR (dB)")
        plt.ylabel("Frame Success Rate")
        plt.title(f"Frame Length: {frame_len} words")

        idx = np.where(mean_frame_success_rate > 0.9)[0]
        sinr_thresh[ww] = median_sinr_all[idx[-1]] if idx.size > 0 else np.nan

    savemat("results.mat", {
        "frameSuccessRateM": frame_success_rate_M,
        "berM": ber_M,
        "sinrThresh": sinr_thresh,
        "frameLenVect": np.array(args.frame_lens),
        "setList": np.array(args.set_list),
    })

    plt.figure()
    plt.plot(args.frame_lens, sinr_thresh, marker="*", linestyle="-")
    plt.grid(True)
    plt.ylabel("SINR Threshold for <10% Frame Error")
    plt.xlabel("Number of Codewords Per Frame")
    plt.title("Performance Evaluation vs. Codewords Per Frame")
    plt.show()


if __name__ == "__main__":
    main()
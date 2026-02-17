import numpy as np

# gRRC: Root Raised Cosine Filter Impulse Reesponse
gRRC = np.array([
    -0.0010719, 0.0065744000000000002, -0.0053055999999999997,
    -0.0020741000000000002, 0.015007, -0.012939000000000001,
    -0.026527999999999999, 0.046175000000000001, 0.037516000000000001,
    -0.12043, -0.045426000000000001, 0.43972, 0.75546999999999997,
    0.43972, -0.045426000000000001, -0.12043, 0.037516000000000001,
    0.046175000000000001, -0.026527999999999999, -0.012939000000000001,
    0.015007, -0.0020741000000000002, -0.0053055999999999997,
    0.0065744000000000002, -0.0010719
], dtype=np.float64)

def read_data(input_filename, Nr, ext_str):
    # Build filename
    filename = f"{input_filename}.{ext_str}"
    
    try:
        # Read float32 little-endian
        full_vect = np.fromfile(filename, dtype='<f4')
    except Exception as e:
        raise IOError(f"Could not read from filename {filename}") from e

    print(f"Finished reading IQ file {filename}")

    # Convert interleaved real/imag into complex
    vect_complex = full_vect[0::2] + 1j * full_vect[1::2]

    # Compute samples per antenna
    total_samples = len(vect_complex)
    if total_samples % Nr != 0:
        raise ValueError("Number of samples is not a multiple of Nr")

    samp_per_ant = total_samples // Nr

    # Reshape exactly like MATLAB (column-major!)
    Y = np.reshape(vect_complex, (samp_per_ant, Nr), order='F')

    # Transpose to match MATLAB output
    Y = Y.T

    return Y

def write_data(output_filename, sig, input_struct):
    """
    output_filename: base name without extension
    sig: (Nr, Nsamples) complex ndarray
    input_struct: dict-like with key 'Nr'
    Writes: output_filename + '.iqdata' as float32 little-endian interleaved IQ.
    """

    if sig is None or sig.size == 0:
        raise ValueError("No data to write")

    Nr_expected = int(input_struct["Nr"])
    if sig.shape[0] != Nr_expected:
        raise ValueError(
            f"User-specified number of receive antennas is {Nr_expected}. "
            f"Expecting data matrix to have {Nr_expected} rows"
        )

    # Ensure MATLAB-like default precision
    sig = np.asarray(sig, dtype=np.complex128)

    # temp = sig.'  (transpose WITHOUT conjugation)
    # In NumPy, .T is also non-conjugating; conjugate transpose would be .conj().T
    temp = sig.T  # shape: (Nsamples, Nr)

    # MATLAB temp(:) stacks columns => Fortran order flatten
    z = temp.reshape(-1, order="F")  # length Nsamples*Nr, complex

    # Build [Re1, Im1, Re2, Im2, ...] as float32 little-endian
    interleaved = np.empty(z.size * 2, dtype=np.float32)
    interleaved[0::2] = z.real.astype(np.float32)
    interleaved[1::2] = z.imag.astype(np.float32)

    # Write little-endian float32
    # np.float32 is native-endian; to force little-endian use '<f4'
    interleaved.astype("<f4").tofile(output_filename + ".iqdata")


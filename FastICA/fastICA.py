import numpy as np

def complex_fastica(xin: np.ndarray, n: int, eps: float = 0.1, n_iter_max: int = 40,
                    tol: float = 1e-3, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Port of MATLAB complexFastICA.m

    Parameters
    ----------
    xin : np.ndarray, shape (p, m)
        Complex input data where:
          p = number of sensors (channels / antennas)
          m = number of samples
        This matches MATLAB usage: m = size(xin,2); p = size(xin,1)
    n : int
        Number of independent components to estimate (<= p).
    eps : float
        Epsilon used in the nonlinearity g = 1/(eps + |w^H x|^2).
    n_iter_max : int
        Max iterations per component.
    tol : float
        Stopping threshold analogous to: sum(abs(abs(wprev) - abs(w))) > 0.001
    rng : np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    sigHat : np.ndarray, shape (n, m)
        Estimated independent components: W^H x_whitened
    """
    xin = np.asarray(xin, dtype=np.complex128)
    p, m = xin.shape
    if n > p:
        raise ValueError(f"n ({n}) cannot exceed number of sensors p ({p}).")

    if rng is None:
        rng = np.random.default_rng()

    # ---- Whitening (matches MATLAB)
    # MATLAB: cov(xin') where xin' is m-by-p => covariance across sensors (p-by-p)
    # We'll compute covariance across sensors directly:
    # cov = (Xc @ Xc^H) / (m-1)   where Xc is mean-centered across time
    Xc = xin - xin.mean(axis=1, keepdims=True)
    cov = (Xc @ Xc.conj().T) / (m - 1)

    # MATLAB: [Ex, Dx] = eig(cov(...));
    # Note: cov is Hermitian -> use eigh for numerical stability
    eigvals, Ex = np.linalg.eigh(cov)  # eigvals ascending
    # Guard against tiny negative due to numerical error
    eigvals = np.maximum(eigvals, 0.0)

    # MATLAB: R = sqrt(inv(Dx)) * Ex'
    # Dx is diagonal of eigvals. sqrt(inv(Dx)) = diag(1/sqrt(eigvals))
    # Ex' in MATLAB is conjugate-transpose; in NumPy: Ex.conj().T
    inv_sqrt = np.zeros_like(eigvals)
    # avoid divide-by-zero for rank-deficient covariance
    mask = eigvals > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(eigvals[mask])
    R = (inv_sqrt[:, None] * Ex.conj().T)  # diag(inv_sqrt) @ Ex^H

    x = R @ xin  # whitened data, shape (p, m)

    # ---- ICA iterations
    W = np.zeros((p, n), dtype=np.complex128)

    for k in range(n):
        # MATLAB: w = rand(p,1) + i*rand(p,1)
        w = rng.random((p,)) + 1j * rng.random((p,))
        w = w / np.linalg.norm(w)

        wprev = np.zeros((p,), dtype=np.complex128)
        iter_count = 0

        while iter_count < n_iter_max and np.sum(np.abs(np.abs(wprev) - np.abs(w))) > tol:
            wprev = w.copy()

            # y = w^H x  (1-by-m)
            y = w.conj().T @ x  # shape (m,)

            # g and dg (both real-valued arrays because they depend on |y|^2)
            abs2 = np.abs(y) ** 2
            g = 1.0 / (eps + abs2)
            dg = -1.0 / (eps + abs2) ** 2

            # MATLAB:
            # w = mean(x .* (ones(p,1)*conj(w'*x)) .* (ones(p,1)*g), 2) - ...
            #     mean(g + abs(w'*x).^2 .* dg) * w;
            #
            # In vector form:
            # term1 = mean( x * conj(y) * g , over samples )
            # term2 = mean( g + |y|^2 * dg )
            term1 = np.mean(x * (np.conj(y) * g)[None, :], axis=1)  # shape (p,)
            term2 = np.mean(g + abs2 * dg)  # scalar (real)
            w = term1 - term2 * w

            # normalize
            w = w / np.linalg.norm(w)

            # decorrelate vs previous components: w = w - W W^H w
            if k > 0:
                Wk = W[:, :k]
                w = w - Wk @ (Wk.conj().T @ w)
                w = w / np.linalg.norm(w)

            iter_count += 1

        W[:, k] = w

    # MATLAB: sigHat = W' * x  (W' is conjugate transpose)
    sigHat = W.conj().T @ x  # shape (n, m)
    return sigHat
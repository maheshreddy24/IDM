""" Visualisation helpers (PCA of patch features -> RGB, a grid plot) and a closed-form linear (ridge) probe. """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .data import IMAGENET_MEAN, IMAGENET_STD


def denormalize(images):
    """ ImageNet-normalised bs, 3, H, W -> bs, H, W, 3 in [0, 1] """
    return (images * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(0, 2, 3, 1)


def fit_pca(feats, k=3):
    """ feats: M, C -> (mean C, components C, k, lo k, hi k); lo / hi are the 1st / 99th percentiles of the
    projections, used to map every set of features projected with this basis onto the same colours """
    mean = feats.mean(0)
    _, _, V = torch.pca_lowrank(feats - mean, q=k, center=False)
    proj = (feats - mean) @ V
    return mean, V, torch.quantile(proj, 0.01, dim=0), torch.quantile(proj, 0.99, dim=0)


def pca_rgb(feats, pca, grid, size=None):
    """ feats: bs, N, C projected with a fit_pca basis -> bs, H, W, 3 in [0, 1] (N = grid[0] * grid[1]);
    upsampled (nearest) to size x size if given """
    mean, V, lo, hi = pca
    rgb = (((feats - mean) @ V - lo) / (hi - lo)).clamp(0, 1)  # bs, N, 3
    rgb = rgb.unflatten(1, grid).permute(0, 3, 1, 2)  # bs, 3, h, w
    if size is not None:
        rgb = F.interpolate(rgb, size=(size, size), mode='nearest')
    return rgb.permute(0, 2, 3, 1)


def plot_grid(images, path, row_titles=None, col_titles=None, size=2.0):
    """ images: R, C, H, W, 3 in [0, 1] -> an R x C grid of panels, saved to path """
    R, C = images.shape[:2]
    fig, axes = plt.subplots(R, C, figsize=(C * size, R * size), squeeze=False)
    for i in range(R):
        for j in range(C):
            axes[i, j].imshow(images[i, j])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0 and col_titles:
                axes[i, j].set_title(col_titles[j], fontsize=9)
            if j == 0 and row_titles:
                axes[i, j].set_ylabel(row_titles[i], fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def fit_ridge(X, Y, ratios=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100), chunk=16384):
    """ Linear probe Y ~ X W + b by ridge regression. X: N, D features (float16 is fine, N << D, e.g. flattened patch
    tokens); Y: N, K targets. Solved in the dual: alpha = (G + lam I)^-1 Yc with G = Xc Xc^T, W = Xc^T alpha, so only
    N x N is ever inverted and X is read in D-chunks. lam = ratio * mean eigenvalue of G, ratio picked by the exact
    leave-one-out error (from one eigendecomposition of G).
    -> {'W': D, K, 'b': K, 'ratio', 'loo_mse'}; predict with ridge_predict """
    N, D = X.shape
    mu = torch.cat([X[:, j:j + chunk].float().mean(0) for j in range(0, D, chunk)])
    G = torch.zeros(N, N, dtype=torch.float64, device=X.device)
    for j in range(0, D, chunk):
        xc = X[:, j:j + chunk].float() - mu[j:j + chunk]
        G += (xc @ xc.T).double()
    y_mu = Y.mean(0)
    Yc = (Y - y_mu).double()
    evals, U = torch.linalg.eigh(G)
    evals = evals.clamp_min(0)
    UtY = U.T @ Yc
    best = None
    for ratio in ratios:
        shrink = evals / (evals + ratio * evals.mean())  # eigenvalues of the hat matrix H = G (G + lam I)^-1
        residual = Yc - U @ (shrink[:, None] * UtY)
        h = (U ** 2) @ shrink + 1 / N  # diag of the full hat matrix, including the intercept's 1 / N
        loo = (residual / (1 - h)[:, None]).pow(2).mean().item()  # LOO residual = residual / (1 - H_ii)
        if best is None or loo < best[1]:
            best = (ratio, loo)
    alpha = (U @ (UtY / (evals + best[0] * evals.mean())[:, None])).float()  # N, K
    W = torch.cat([(X[:, j:j + chunk].float() - mu[j:j + chunk]).T @ alpha for j in range(0, D, chunk)])  # D, K
    return {'W': W, 'b': y_mu.float() - mu @ W, 'ratio': best[0], 'loo_mse': best[1]}


@torch.no_grad()
def ridge_predict(probe, X, bs=256):
    """ X: N, D -> N, K """
    return torch.cat([X[i:i + bs].float() @ probe['W'] + probe['b'] for i in range(0, len(X), bs)])

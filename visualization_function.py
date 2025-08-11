import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.cross_decomposition import CCA
from scipy import stats
from sklearn.decomposition import PCA

def cal_likelihood(n_MT, n_WT, n_missing, theta, p_obs_MT, p_obs_WT):
    """Calculate log-likelihood for given counts and theta."""
    return (
        n_MT * np.log(theta)
        + n_WT * np.log(1 - theta)
        + n_missing * np.log(1 - p_obs_MT * theta - p_obs_WT * (1 - theta))
    )

def cal_prob(matrix, p_obs_MT, p_obs_WT):
    """
    Estimate theta for each position using maximum likelihood.
    
    Parameters:
        matrix: np.ndarray of shape (n_samples, n_positions)
                1 = MT, -1 = WT, 0 = missing
        p_obs_MT: prior observation probability for MT
        p_obs_WT: prior observation probability for WT
    
    Returns:
        consensus: np.ndarray of estimated theta values per position
    """
    n_positions = matrix.shape[1]
    consensus = np.zeros(n_positions)

    for j in range(n_positions):
        n_MT = (matrix[:, j] == 1).sum()
        n_WT = (matrix[:, j] == -1).sum()
        n_missing = (matrix[:, j] == 0).sum()

        if n_MT > 0 or n_WT > 0:
            result = minimize_scalar(
                lambda x: -cal_likelihood(n_MT, n_WT, n_missing, x, p_obs_MT, p_obs_WT),
                bounds=(0.01, 0.99),
                method='bounded'
            )
            consensus[j] = result.x
        else:
            consensus[j] = np.nan  # no data at this position

    return consensus

def mutation_likelihood(parent_consensus, child_consensus, p=0.95, q=0.2):
    """
    Calculate the log-likelihood of a child's mutation profile given a parent's profile.

    Parameters:
        parent_consensus (array-like): Parent consensus calls (0 or 1).
        child_consensus (array-like): Child consensus calls (0 or 1).
        p (float): Probability of child matching parent when parent has mutation (1).
        q (float): Probability of child matching parent when parent has wildtype (0).

    Returns:
        float: Log-likelihood value.
    """
    # Convert inputs to numpy arrays
    parent = np.asarray(parent_consensus)
    child = np.asarray(child_consensus)

    # Ensure shapes match
    if parent.shape != child.shape:
        raise ValueError("Parent and child consensus arrays must have the same shape.")

    # Count combinations
    c11 = np.sum((parent == 1) & (child == 1))
    c01 = np.sum((parent == 0) & (child == 1))
    c10 = np.sum((parent == 1) & (child == 0))
    c00 = np.sum((parent == 0) & (child == 0))

    # Compute log-likelihood
    log_prob = (
        c11 * np.log(p) +
        c01 * np.log(1 - q) +
        c10 * np.log(1 - p) +
        c00 * np.log(q)
    )

    return log_prob

def CCA_projection(feature_matrix, x):
    """
    Project a variable x onto the first canonical component of feature_matrix using CCA.

    Parameters:
        feature_matrix (array-like): Shape (n_samples, n_features), the explanatory features.
        x (array-like): Shape (n_samples,) or (n_samples, 1), the target variable.

    Returns:
        np.ndarray: Normalized projection of x, scaled to [0, 1].
    """
    feature_matrix = np.asarray(feature_matrix)
    x = np.asarray(x).reshape(-1, 1)  # Ensure column vector

    if feature_matrix.shape[0] != x.shape[0]:
        raise ValueError("feature_matrix and x must have the same number of samples.")

    # CCA projection (first canonical component)
    project_x, _ = CCA(n_components=1).fit_transform(feature_matrix, x)
    project_x = project_x.ravel()

    # Align sign with Spearman correlation
    rho, _ = stats.spearmanr(project_x, x.ravel())
    project_x *= np.sign(rho)

    # Normalize to [0, 1]
    if np.ptp(project_x) == 0:  # Avoid division by zero
        project_x = np.zeros_like(project_x)
    else:
        project_x = (project_x - project_x.min()) / (project_x.max() - project_x.min())

    return project_x

def guided_residual_projection(X_2d, y):
    """
    Reorients a 2D embedding X_2d by:
    - First axis: aligned with y
    - Second axis: residual direction orthogonal to y, with max variance

    Parameters:
        X_2d: [n_samples, 2] numpy array
        y:    [n_samples] 1D numpy array

    Returns:
        proj_2d: [n_samples, 2] numpy array (aligned projection)
    """
    # Center X and y
    X = X_2d - X_2d.mean(axis=0, keepdims=True)
    y = y - y.mean()
    y = y.reshape(-1, 1)

    # Axis 1: best linear direction in X that predicts y
    w1 = np.linalg.lstsq(X, y, rcond=None)[0].flatten()  # shape (2,)
    w1 /= np.linalg.norm(w1)

    # Project X onto w1 to get the "y-direction"
    x1 = X @ w1

    # Remove that component from X to get residual
    X_resid = X - np.outer(x1, w1)

    # Axis 2: direction of max variance in residual
    pca = PCA(n_components=1)
    x2 = pca.fit_transform(X_resid).flatten()

    # Return aligned 2D projection
    proj_2d = np.stack([x1, x2], axis=1)
    return proj_2d

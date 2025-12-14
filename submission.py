from typing import Tuple
import numpy as np
from einops import einsum, rearrange

############################################################
# Problem 1


def linear_project(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a batched linear projection using einsum from einops.

    Shapes:
    - x: (batch, d_in)
    - W: (d_in, d_out)
    - b: (d_out,)
    Returns:
    - y: (batch, d_out) where y = x * W + b

    Implementation notes:
    - Use einsum from einops for the matrix multiplication (no @ operator).
    - Use NumPy broadcasting for the bias addition.
    - No Python loops.
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    return np.einsum('ij, jk -> ik' , x , W) + b
    # return x.dot(W) + b
    # END_YOUR_CODE


def split_last_dim_pattern() -> str:
    """
    Return an einops.rearrange pattern string that reshapes
    (B, D) -> (B, G, D/G), where G is provided as a keyword
    (e.g., g=num_groups) when applying rearrange.

    I.e., your returned string `pattern`would be used as follows:
    >>> y = rearrange(x, pattern, g=num_groups)
    where x is a tensor with shape (B, D).
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    return 'b (g c) -> b g c'
    # END_YOUR_CODE


def normalized_inner_products(A: np.ndarray, C: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Batched all-pairs dot products, optionally scaled by 1/sqrt(d).

    Board-style intro:
    - Let A in R^{B x M x D} and C in R^{B x N x D}.
    - For each batch b, form S[b,i,j] = <A[b,i,:], C[b,j,:]> giving S in R^{B x M x N}.
    - If normalize is True, scale S by 1/sqrt(D) to keep magnitudes comparable across D.

    Shapes:
    - A: (batch, m, d)
    - C: (batch, n, d)
    Returns:
    - S: (batch, m, n) with S[b,i,j] = <A[b,i], C[b,j]>.

    Implementation notes:
    - Use einsum from einops for the contraction (no @ operator or loops).
    - Think about the Einstein notation pattern for batched dot products.
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    num = np.einsum('ijk ,ilk->ijl',A,C)
    if normalize:
       return num /  np.sqrt(A.shape[-1])
    return num
    
    # END_YOUR_CODE


def mask_strictly_upper(scores: np.ndarray) -> np.ndarray:
    """
    Mask strictly upper-triangular entries (j > i) to -np.inf via broadcasting.

    Board-style intro:
    - Let scores in R^{B x L x L} be a batch of square matrices.
    - For each matrix, we want to set entries with column index j greater than row index i to -inf.
    - Construct the boolean mask using broadcasted index grids; avoid loops.

    Shapes:
    - scores: (batch, L, L)
    Returns:
    - masked_scores: (batch, L, L) where entries with column > row are -inf.

    Use NumPy broadcasting to construct and apply the mask without loops.
    Note that the data type should be floats.
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement 
    L  = scores.shape[1]
    matrix = -np.inf * np.ones((L,L))
    upper = np.triu(matrix,k=1)
    upper = scores + upper
    return upper
    # END_YOUR_CODE


def prob_weighted_sum_einsum() -> str:
    """
    Batch probability-weighted sums over value vectors using einsum.

    Board-style intro:
    - Let P in R^{B x N} be per-batch probability weights (each row sums to 1).
    - Let V in R^{B x N x D} be per-batch value vectors aligned with P along N.
    - Compute out[b,:] = sum_{j=1..N} P[b,j] * V[b,j,:] in R^{D} for each batch b.

    Return a einops.einsum string that computes out = sum_j P[b,j] * V[b,j,:]
    for shapes P:(B,N), V:(B,N,D) -> out:(B,D).

    I.e., your returned string `pattern` would be used as follows:
    >>> out = einops.einsum(P, V, pattern)
    where P is a tensor with shape (B, N) and V is a tensor with shape (B, N, D).
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    
    return 'i j ,i j k->i k'
    # END_YOUR_CODE


############################################################
# Problem 2


def gradient_warmup(w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of f(w) = sum_i (w_i - c_i)^2 with respect to w.

    Inputs:
    - w: (d,)
    - c: (d,)
    Returns:
    - grad: (d,)
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    return 2 * (w - c)
    # END_YOUR_CODE


def matrix_grad(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For s = sum_{i,j} (A B)_{i,j}, compute gradients wrt A and B.

    If A is (m, p) and B is (p, n):
    - grad_A[i, k] = sum_j B[k, j]  (independent of i)
    - grad_B[k, j] = sum_i A[i, k]  (independent of j)

    Returns (grad_A, grad_B) with the same shapes as A and B, respectively.

    Implementation notes:
    - Consider using einsum from einops for computing sums over dimensions.
    - Alternatively, NumPy sum operations are acceptable for this problem.
    - Use broadcasting to replicate values to the correct shapes.
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    one_A = np.ones((A.shape[0],1))
    one_B = np.ones((1,B.shape[1]))
    grad_A = np.einsum('ij->i' ,B).reshape(1,-1)
    grad_B = np.einsum('ij->j' ,A).reshape(-1,1)
    grad_A = one_A.dot(grad_A)
    grad_B = grad_B.dot(one_B)
    return (grad_A , grad_B)
    # END_YOUR_CODE


def lsq_grad(w: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Analytic gradient for f(w) = 1/2 * ||A w - b||_2^2 with respect to w.

    Inputs:
    - w: (d,)
    - A: (n, d)
    - b: (n,)
    Returns:
    - grad: (d,) = A^T (A w - b)

    Implementation notes:
    - Consider using einsum from einops for matrix-vector operations.
    - Alternatively, NumPy @ operator is acceptable here for simplicity.
    - No Python loops.
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    return A.T @ (A @ w - b)
    # END_YOUR_CODE


def lsq_finite_diff_grad(w: np.ndarray,
                         A: np.ndarray,
                         b: np.ndarray,
                         epsilon: float = 1e-5) -> np.ndarray:
    """
    Central-difference numerical gradient for f(w) = 1/2 * ||A w - b||_2^2.

    Inputs:
    - w: (d,)
    - A: (n, d)
    - b: (n,)
    - epsilon: small step size for finite differences
    Returns:
    - grad_fd: (d,)

    Implementation notes:
    - Compute each component using central differences.
    - Vectorize the computation using NumPy broadcasting.
    - Consider using einsum from einops for matrix-vector operations.
    - No Python loops over gradient components.
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    grad_fd = np.zeros(w.shape)
    w = w.reshape(-1,1)
    b = b.reshape(-1,1)
    for i in range(w.shape[0]):
        e = np.eye(w.shape[0],1)
        e[0] =0
        e[i] = epsilon
        add_ = w + e
        minus_ = w - e
        fw = A @ add_
        fw = fw - b
        fw = fw * fw
        fw = 1/2 * np.sum(fw)
        fd = A @ minus_
        fd = fd - b
        fd = fd * fd
        fd = 1/2 * np.sum(fd)
        grad_fd[i] = (fw - fd) / (2 * epsilon)
    return grad_fd
    # END_YOUR_CODE
############################################################
# Problem 3c


def gradient_descent_quadratic(x: np.ndarray, w: np.ndarray, theta0: float, lr: float,
                               num_steps: int) -> float:
    """
    Minimize f(θ) = sum_i w_i * (θ - x_i)^2 with gradient descent in 1D.

    Inputs:
    - x: (n,) data values
    - w: (n,) positive weights
    - theta0: initial scalar θ
    - lr: learning rate (stepsize)
    - num_steps: number of gradient steps (non-negative integer)

    Returns:
    - theta: final scalar after num_steps updates

    Gradient: df/dθ = 2 * sum_i w_i * (θ - x_i).
    """
    # BEGIN_YOUR_CODE
    # TODO: Implement
    theta = theta0
    for i in range(num_steps):
        theta = theta - lr * 2 * (w.dot(theta - x))
    return theta
    # END_YOUR_CODE

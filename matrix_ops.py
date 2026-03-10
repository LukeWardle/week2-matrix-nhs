"""
matrix_ops.py
Core matrix operations for Week 2 of AI Engineering Course.

Implements: addition, scalar multiplication, matrix multiplication,
transpose, identity, inverse and determinate operations.

Mathematical foundation: Module 1 - Linear Algebra for AI Engineering
Author: Luke Wardle
Date: 10/03/2026

"""

import numpy as np
from typing import Union, Optional

def matrix_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Add two matrices element-wise.

  Mathematically: (A + B)[i, j] = A[i, j] + B[i, j]
  Both matrices must have identical dimensions (m x n).

  Args:
    A: First matrix (m x n)
    B: Second matrix (m x n, must match A's shape)

  Returns:
    Result matrix (m x n)

  Raises:
    ValueError: If matrices have different shapes
  
  """
  if A.shape != B.shape:
    raise ValueError(
      f"Cannot add matrices with shapes {A.shape} and {B.shape}."
      f"Matrices must have identical dimensions."
    )
  return A + B

def scalar_multiply(scalar: float, A: np.ndarray) -> np.ndarray:
  """
  Multiply a matrix by a scalar value.

  Mathematically: (c * A)[i, j] = c * A[i, j]
  This implements the homogeneity property of linear transformations.

  Args:
    scalar: The scalar multiplier (any real number)
    A: Matrix to scale (m x n)

  Returns:
    Scaled matrix (m x n)
  
  """
  return scalar * A

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Multiply two matrices: C = A @ B
  Mathematically: C[i, j] = sum over k of A[i, k] * B[k, j]
  Requires: A is (m x n), B is (n x p) -> result is (m x p).
  The inner dimensions must match: A.shape[1] == B.shape[0].

  Args:
    A: First matrix (m x n)
    B: Second matrix (n x p)

  Returns:
    Result matrix (m x p)

  Raises:
    ValueError: If inner dimensions do not match
  
  """
  if A.shape[1] != B.shape[0]:
    raise ValueError(
      f"Cannot multiply: A is {A.shape}, B is {B.shape}."
      f"A's columns {{A.shape[1]}} must equal B's rows {{B.SHAPE[0]}}."
    )
  return A @ B

def matrix_transpose(A: np.ndarray) -> np.ndarray:
  """
  Transpose a matrix: swap rows and columns.

  Mathematically: (Aᵀ)[i,j] = A[j,i]
  An (m x n) matrix transpose to (n x m).

  Args:
    A: Matrix (m x n)

  Returns:
    Transposed matrix (n x m)
  
  """
  return A.T

def create_identity(n: int) -> np.ndarray:
  """
  Create an n x n identity matrix.

  The identity matrix has 1s on the main diagonal, 0s elsewhere.
  For any matrix A: A @ I = I @ A = A (identity of multiplication).

  Args:
    n: Size of the square identity matrix

  Returns:
    Identity matrix (n x n)

  Raises:
    ValueError: If n is not a positive integer
  
  """
  if n < 1:
    raise ValueError(f"Identity matrix size must be positive, got {n}")
  
  return np.eye(n)

def matrix_inverse(A: np.ndarray) -> Optional[np.ndarray]:
  """
  Compute the inverse of a square matrix if it exists.

  A matrix A is invertible if det(A) != 0. The inverse A_inv
  satisfies: A @ A_inv = A_inv @ A = I (the identity matrix).

  Near-singular matrices (|det| < 1e-10) are treated as singular
  because floating-point arithmetic makes their 'inverse' numerically
  unreliable.

  Args:
    A: Square matrix (n x n)

  Returns:
    Inverse matrix (n x n), or None if matrix is singular.

  Raises:
    ValueError: If A is not square
  
  """
  if A.shape[0] != A.shape[1]:
    raise ValueError(
      f"Matrix must be square to have an inverse."
      f"Got shape {A.shape}."
    )
  
  det = np.linalg.det(A)

  if np.abs(det) < 1e-10:
    print(f"Warning: Matrix is singular (det = {det:.2e})."
          f"No inverse exists.")
    return None
  
  return np.linalg.inv(A)

def matrix_determinant(A: np.ndarray) -> float:
  """
  Compute the determinant of a square matrix.

  The determinant measures:
  - |det(A)|: area/volume scaling factor of the transformation
  - sign(det(A)): orientation (+ preserved, - reversed)
  - det(A) == 0: matrix is singular (non-invertible)

  Args:
    A: Square matrix (n x n)

  Returns:
    Scalar determinant value

  Raises:
    ValueError: If A is not square
  
  """
  if A.shape[0] != A.shape[1]:
    raise ValueError(
      f"Determinant only defined for square matrices."
      f"Got shape {A.shape}."
    )
  return float(np.linalg.det(A))
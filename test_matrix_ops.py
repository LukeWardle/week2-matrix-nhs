"""
test_matrix_ops.py
Comprehensive test suite for matrix_ops module.

Each test verifies a specific mathematical property. Tests are independent - they do not rely on each other's
results.

"""

import numpy as np
import matrix_ops as mo

def test_matrix_add():
  """Test element-wise addition: (A+B)[i, j] = A[i, j] + B[i, j]."""
  A = np.array([[1, 2], [3, 4]])
  B = np.array([[5, 6], [7, 8]])
  result = mo.matrix_add(A, B)
  expected = np.array([[6, 8], [10, 12]])
  assert np.array_equal(result, expected), (
    f"Matrix addition failed. Got {result}, expected {expected}"
  )
  print("  ✓ matrix_add: element-wise addition correct")

def test_matrix_add_shape_error():
  """Test that adding incompatible shapes raises ValueError."""
  A = np.array([[1, 2], [3, 4]])
  B = np.array([[1, 2, 3], [4, 5, 6]])
  try:
    mo.matrix_add(A, B)
    assert False, "Should have raised a ValueError"
  except ValueError:
    print(" ✓ matrix_add: shape mismatch correctly raises ValueError")

def test_scalar_multiply():
  """Test that scalar_multiply scales every element uniformly."""
  A = np.array([[1, 2], [3, 4]])
  result = mo.scalar_multiply(3, A)
  expected = np.array([[3, 6], [9, 12]])
  assert np.array_equal(result, expected), (
    f"Scalar multiply failed. Got {result}, expected {expected}"
  )
  print("  ✓ scalar_multiply: uniform scaling correct")

def test_matrix_multiply():
  """Test matrix multiplication: C[i, j] = sum_k A[i, k]*B[k, j]."""
  A = np.array([[1, 2], [3, 4]])
  B = np.array([[2, 0], [1, 3]])
  result = mo.matrix_multiply(A, B)
  # A @ B: row 0 = [1*2+2*1, 1*0+2*3] = [4, 6] row 1 = [3*2+4*1, 3*0+4*3] = [10, 12]
  expected = np.array([[4, 6], [10, 12]])
  assert np.array_equal(result, expected), (
    f"Matrix multiply failed. Got {result}, expected {expected}"
  )
  print("  ✓ matrix_multiply: matrix product correct")

def test_matrix_multiply_non_commutative():
  """Test that A @ B != B @ A in general (non-commutativity)."""
  A = np.array([[1, 2], [0, 1]])
  B = np.array([[1, 0], [3, 1]])
  AB = mo.matrix_multiply(A, B)
  BA = mo.matrix_multiply(B, A)
  assert not np.array_equal(AB, BA), "AB should not equal BA"
  print("  ✓ matrix_multiply: non-commutativity confirmed (AB != BA)")

def test_matrix_transpose():
  """Test transpose: (Aᵀ)[i,j] = A[j,i]."""
  A = np.array([[1, 2, 3], [4, 5, 6]])
  result = mo.matrix_transpose(A)
  expected = np.array([[1,4], [2, 5], [3, 6]])
  assert np.array_equal(result, expected), (
    f"Transpose failed. Got {result}, expected {expected}"
  )
  print("  ✓ matrix_transpose: row/column swap correct")

def test_identity():
  """Test that create_identity produces correct I and A @ I = A."""
  I = mo.create_identity(3)
  expected_I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  assert np.array_equal(I, expected_I), "Identity matrix wrong"

  # Verify A @ I = A for a test matrix
  A = np.array([[2, 5], [1, 3]])
  I2 = mo.create_identity(2)
  result = mo.matrix_multiply(A, I2)
  assert np.array_equal(result, A), "A @ I should equal A"
  print("  ✓ create_identity: I correct, and A @ I = A verified")

def test_matrix_inverse():
  """Test inverse: A @ A_inv = I (to floating-point precision)."""
  A = np.array([[4, 7], [2, 6]])
  A_inv = mo.matrix_inverse(A)
  assert A_inv is not None, "Invertible matrix returned None"

  # Verify A @ A_inv = I
  product = mo.matrix_multiply(A, A_inv)
  I = np.eye(2)
  # Use np.allclose, not np.array_equal: floating-point arithmetic
  # means entries may be 0.99999999 rather than exactly 1
  assert np.allclose(product, I, atol=1e-10), (
    f"Inverse verification failed. A @ A_inv = {product}"
  )
  print("  ✓ matrix_inverse: A @ A_inv = I verified")

def test_singular_matrix():
  """Test that singular matrix (det=0) returns None."""
  # Second row = 2 * first row -> linearly dependent -> singular
  A = np.array([[1, 2], [2, 4]])
  result = mo.matrix_inverse(A)
  assert result is None, (
    "Singular matrix should return None, not an inverse"
  )
  print("  ✓ matrix_inverse: singular matrix correctly returns None")

def test_determinant_properties():
  """Test key determinant properties: identity=1, rotation=1."""
  # det(I) = 1
  I = mo.create_identity(3)
  assert np.isclose(mo.matrix_determinant(I), 1.0), "det(I) should be 1"

  # det(R) = 1 for 90-degree rotation
  R = np.array([[0, -1], [1, 0]])
  assert np.isclose(mo.matrix_determinant(R), 1.0), "det(R90) should be 1"

  # det = 0 for singular matrix
  S = np.array([[2, 4], [1, 2]])
  assert np.isclose(mo.matrix_determinant(S), 0.0), "det(singular) should be 0"

  print("  ✓ matrix_determinant: det(I)=1, det(R)=1, det(singular)=0")

if __name__ == "__main__":
   print("\n" + "="*55)
   print("  MATRIX OPERATIONS TEST SUITE") 
   print("="*55 + "\n") 
   test_matrix_add() 
   test_matrix_add_shape_error() 
   test_scalar_multiply() 
   test_matrix_multiply() 
   test_matrix_multiply_non_commutative() 
   test_matrix_transpose() 
   test_identity() 
   test_matrix_inverse() 
   test_singular_matrix() 
   test_determinant_properties() 
   print("\n" + "="*55) 
   print("  ALL 10 TESTS PASSED")
   print("="*55 + "\n")  

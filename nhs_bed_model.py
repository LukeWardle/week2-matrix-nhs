"""
nhs_bed_model.py

NHS Regional Bed Allocation Simulator

Demonstrates matrix operations applied to healthcare resource planning.
Models how inter-regional overflow agreements affect total bed load
across NHS regions multiple weeks.

Mathematical model: Adjusted = D @ W^T
Revocery:           D = Adjusted @ (W^T)^(-1)

Author: Luke Wardle
Date: 12/03/2026

"""

import numpy as np
import matrix_ops as mo

def simulate_bed_allocation():
  """
  Simulate NHS regional bed allocation over 3 weeks for 3 regions.

  Returns:
    Tuple: (demand, weights, adjusted_allocation)

  """
  # ── Demand Matrix ──────────────────────────────────────────────── 
  # Shape: (3 weeks) x (3 regions)
  # D[i,j] = beds needed in region j during week i
  demand = np.array([
    [120, 85, 100],   # Week 1: Regional=120, Regions2=85, Region3=100
    [110, 90, 105],   # Week 2: Slight demand shift
    [130, 80, 110],   # Week 3: Regional surge, Region2 lower
  ], dtype=float)

  # ── Weights Matrix ─────────────────────────────────────────────── 
  # Shape: (3 regions) x (3 regions)
  # W[i,j] = fraction of region j's demand absorbed by region i
  # Diagonal = 1.0: each region handles 100% of its own demand
  # Off-diagonal: overflow-sharing agreements between NHS Trusts
  weights = np.array([
    [1.00, 0.10, 0.20], # Region 1: absorbs 10% of R2, 20% of R3
    [0.05, 1.00, 0.10], # Region 2: absorbs 5% of R1, 10% of R3
    [0.10, 0.20, 1.00], # Region 3: absorbs 10% of R1, 20% of R2
  ], dtype=float)

  # ── Compute Adjusted Allocation ──────────────────────────────────
  weights_T = mo.matrix_transpose(weights)
  adjusted_allocation = mo.matrix_multiply(demand, weights_T)

  return demand, weights, adjusted_allocation

def display_results(demand, weights, adjusted):
  """Display the simulation results with formatted output tables."""
  print("=" * 62)
  print(" NHS REGIONAL BED ALLOCATION SIMULATION")
  print("=" * 62)

  # Original demand
  print("\n Regional Demand (beds needed per week):")
  print(f"  {'':10s} {'Region 1':>10s} {'Region 2':>10s} {'Region 3':>10s}")

  for week_idx, row in enumerate(demand, 1):
    print(f"  Week {week_idx}:  {row[0]:>10.0f} {row[1]:>10.0f} {row[2]:>10.0f}")

  # Weight agreements
  print("\n INTER-REGIONAL SUPPORT WEIGHTS:")
  print(" {W[i,j] = fraction of Region j's demand Region i absorbs}")
  print(f"  {'':12s} {'From R1':>0s} {'From R2':>8s} {'From R3':>8s}")
  for i, label in enumerate(['Region 1', 'Region 2', 'Region 3']):
    print(f" {label}: {weights[i,0]:>8.2f} {weights[i,1]:>8.2f} {weights[i,2]:>8.2f}")

  # Adjusted allocation
  print("\n ADJUSTED ALLOCATION (total load including overflow):")
  print(f"  {'':10s} {'Region 1':>10s} {'Region 2':>10s} {'Region 3':>10s}")

  for week_idx, row in enumerate(adjusted, 1):
    print(f"  Week {week_idx}:  {row[0]:>10.1f} {row[1]:>10.1f} {row[2]:>10.1f}")

  # Overflow
  overflow = adjusted - demand
  print("\n NET OVERFLOW (+ = receiving overflow, - = sending):")
  print(f"  {'':10s} {'Region 1':>10s} {'Region 2':>10s} {'Region 3':>10s}") 
  
  for week_idx, row in enumerate(overflow, 1): 
    print(f"  Week {week_idx}:   {row[0]:>+10.1f} {row[1]:>+10.1f} {row[2]:>+10.1f}") 
 
 
def verify_identity_baseline(demand): 
  """Show that D @ I = D (identity = no sharing).""" 
  print("\n" + "=" * 62) 
  print("  VERIFICATION: Identity Matrix (No Sharing Baseline)") 
  print("=" * 62) 
  I = mo.create_identity(3) 
  no_change = mo.matrix_multiply(demand, I) 
  print(f"\n  Demand @ I = original demand? {np.allclose(demand, no_change)}") 
  print("  (Identity means: each region handles only its own demand.)") 
 
 
def verify_inverse_recovery(demand, weights): 
  """Demonstrate that inverse of W^T recovers original demand.""" 
  print("\n" + "=" * 62) 
  print("  INVERSE RECOVERY: Audit Trail Verification") 
  print("=" * 62) 
 
  weights_T = mo.matrix_transpose(weights) 
  det_W = mo.matrix_determinant(weights) 
  print(f"\n  det(W) = {det_W:.6f}") 
 
  weights_T_inv = mo.matrix_inverse(weights_T) 
 
  if weights_T_inv is not None: 
    adjusted = mo.matrix_multiply(demand, weights_T) 
    recovered = mo.matrix_multiply(adjusted, weights_T_inv) 
    print("\n  Recovered demand (from Adjusted @ (W^T)^-1):") 
    print(np.round(recovered, 2)) 
    print("\n  Original demand:") 
    print(demand) 
    print(f"\n  Recovery successful? {np.allclose(demand, recovered)}") 
    print("  (Required for NHS NICE compliance — audit trail verified.)") 
  else: 
    print("\n  CANNOT RECOVER: Weights matrix is singular!") 
    print("  Audit trail compliance FAILED.") 
 
 
def main(): 
  """Main execution: run simulation and all verifications.""" 
  demand, weights, adjusted = simulate_bed_allocation() 
  display_results(demand, weights, adjusted) 
  verify_identity_baseline(demand) 
  verify_inverse_recovery(demand, weights) 
 
 
if __name__ == "__main__": 
    main()


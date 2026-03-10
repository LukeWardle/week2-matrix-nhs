# Week 2: Matrix Operations for NHS Bed Allocation

## Overview
Implements core matrix operations and applies them to simulate NHS regional bed availability across England.
Built as part of a AI Engineering course, week 2.

## Mathmatical Background
This project implements linear algebra operations that underpin machine learning systems:
- Matrix addition, scalar multiplication, matrix multiplication
- Matrix transpose and inverse computation
- Determinant-based invertibility checking
- Application: inter-regional resource allocation modelling

## Project Structure
```
week2_matrix_nhs/
├── matrix_ops.py       # Core matrix operations module 
├── test_matrix_ops.py  # Comprehensive test suite
├── nhs_bed_model.py    # NHS bed allocation application
├── README.md           # This file

```

## Setup (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install numpy pandas
```

## Usage
```python
import matrix_ops as mo
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = mo.matrix_multiply(A, B)
print(results)  # [[19, 22], [43, 50]]
```

## Running Tests
```bash
python test_matrix_ops.py
```

## Running the NHS Model
```bash
python nhs_bed_model.py
```

## Author
Created as part of AI Engineering Course - Week 2, Module 1
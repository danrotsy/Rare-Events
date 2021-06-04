# Rare Events

## Code

- <code>loss.py</code> has all the loss functions:
  - Linear: $g(\vec{x}) = \vec{a} \cdot \vec{x}$
  - Quadratic: $g(\vec{x}) = \vec{a} \cdot \vec{x} + \vec{x}^T B \vec{x}$
  - Brownian: $g(\vec{x}) = \max_{i} B_i(\vec{x})$
  - New Brownian: $g(\vec{x}) = \max_i b_i * x_i$
- <code>method.py</code> has all the methods used to estimate $\hat{p}_F$:
  - Monte Carlo
  - SVM
  - Random Forest
  - Linear Regression
  - Polynomial Regression
- <code>prob.py</code> has all the methods to used to compute $p_F$ from $\alpha$ and vice versa for: (it is time consuming to compute $\alpha$ from $p_F$ for Quadratic and Brownian functions)
  - Linear Loss
  - New Brownian Loss

## Notebooks

- <code>Problem Formluation.ipynb</code> explains the Rare Events problem
- <code>Generating Tests.ipynb</code> explains the workflow for testing methods on different functions
- <code>Relating pF and alpha.ipynb</code> explains the new Brownian loss functions
- <code>Pipeline.ipynb</code> generates some tests and saves to a CSV
- <code>Filtration.ipynb</code> compiles the results from the pipeline notebook
- <code>Analysis.ipynb</code> graphs the summaries given by the filtration notebook
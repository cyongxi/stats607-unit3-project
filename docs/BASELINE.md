# Baseline Performance Analysis

## 1. Overview

This document evaluates the computational performance and numerical stability of the original Unit 2 simulation study, which reproduces the Lasso signed-support recovery experiments from Wainwright (2009), Figure 2.

The study includes 126 scenarios, with 200 Monte Carlo replications per scenario, for a total of 25,200 Lasso fits.

All baseline measurements were performed before modifying the original simulation implementation.

## 2. Runtime Benchmark

The original simulation was executed three times using:

```bash
/usr/bin/time -p make simulate
```

The measurements were collected on macOS using Python 3.11.1 in a project-specific virtual environment.

| Run | Wall-clock (s) | User CPU (s) | System CPU (s) |
|---|---:|---:|---:|
| 1 | 76.97 | 69.30 | 6.52 |
| 2 | 73.75 | 67.93 | 5.51 |
| 3 | 73.70 | 67.94 | 5.47 |

Summary:

- Mean runtime: 74.81 seconds
- Median runtime: 73.75 seconds
- Minimum runtime: 73.70 seconds
- Maximum runtime: 76.97 seconds
- Interquartile range: approximately 1.64 seconds, using linearly interpolated quartiles

The median runtime of 73.75 seconds is used as the reference for subsequent simulation benchmarks.

These measurements cover the simulation stage (`make simulate`), not the analysis and figure-generation stages. The earlier Unit 2 record reported approximately 74.90 seconds for `make all` in its original environment; that historical measurement is reported separately because the execution scope and environment differ.

## 3. Runtime Profiling

The original full simulation was profiled using Python's `cProfile`:

```bash
python -m cProfile -o results/profiling/baseline_full.prof -m src.simulation
```

The profiler reported 85.146 seconds and approximately 49.05 million function calls.

The main computational components were:

| Function | Calls | Cumulative time (s) |
|---|---:|---:|
| run_simulation | 1 | 82.993 |
| fit_lasso | 25,200 | 49.630 |
| generate_data | 25,200 | 30.892 |
| exact_recovery | 25,200 | 0.282 |
| calculate_lambda | 25,200 | 0.042 |

Lasso fitting accounted for approximately 59.8% of the simulation function's cumulative runtime, while data generation accounted for approximately 37.2%.

Together, these components represented approximately 97% of the measured simulation runtime.

Within Lasso fitting, substantial time was spent in scikit-learn's coordinate-descent implementation and data validation routines.

The dominant self-time components included:

| Component | Self time (s) |
|---|---:|
| generate_data | 30.621 |
| enet_path | 19.888 |
| numpy.asarray | 10.794 |

These results identify Lasso fitting and Gaussian data generation as the primary optimization targets.

A separate pilot profiling run, using five replications, showed that Python module imports dominated the much smaller workload. This indicates that representative simulation sizes are necessary when identifying performance bottlenecks.

## 4. Computational Complexity

An empirical complexity study was conducted with the following configuration:

- Number of predictors: p = 256
- Sparsity level: k = 103
- Sample sizes: n = 100, 200, 400, 800, 1600
- Timing repetitions: 10 per sample size
- Random seed: 6072026

The experiment separately measured data generation, Lasso fitting, and their combined runtime.

The empirical scaling exponent was estimated by fitting:

\[
\log T(n) = a + b\log n.
\]

Results:

| Component | Estimated exponent |
|---|---:|
| Data generation | 0.968 |
| Lasso fitting | 0.406 |
| Combined runtime | 0.585 |

### 4.1 Data Generation

The original data generator creates an n-by-p Gaussian design matrix and calculates the response using a matrix-vector product.

The main operations have computational cost proportional to O(np). With p fixed, the expected scaling with respect to n is approximately linear.

The estimated exponent of 0.968 is consistent with this prediction over the tested range.

### 4.2 Lasso Fitting

The observed Lasso fitting times were not monotonic in sample size.

In particular, the median fitting time decreased from approximately 2.754 ms at n = 800 to 1.436 ms at n = 1600.

Lasso uses an iterative coordinate-descent solver, and its runtime depends on more than the dimensions of the design matrix. Convergence behavior, regularization strength, and numerical properties of the generated data may influence the number of required iterations.

The regularization parameter also changes with n:

\[
\lambda_n =
\sqrt{\frac{2\sigma^2\log(k)\log(p-k)}{n}}.
\]

Therefore, the estimated exponent of 0.406 describes the empirical behavior of this particular experiment and should not be interpreted as a general theoretical complexity bound for Lasso.

The smallest tested sample size, n = 100, is below k = 103; this configuration is retained as part of the computational scaling experiment, not as a representative support-recovery regime.

The baseline complexity visualization is saved at:

`results/figures/baseline_complexity.png`

## 5. Numerical Stability

### 5.1 Full Simulation Diagnostics

The saved results from all 25,200 replications were examined using the number of coordinate-descent iterations and the reported duality gap.

| Diagnostic | Result |
|---|---:|
| Missing iteration counts | 0 |
| Missing duality gaps | 0 |
| Replications reaching max_iter = 20,000 | 0 |
| Non-finite duality gaps | 0 |
| Negative reported duality gaps | 1 |
| Maximum iteration count | 106 |
| Median iteration count | 12 |
| 99th percentile iteration count | 49 |

The median duality gap was approximately 4.27 × 10⁻⁸, and the maximum was approximately 3.43 × 10⁻⁶.

One negative reported duality gap of approximately −1.15 × 10⁻¹⁶ was observed. Its magnitude is consistent with floating-point rounding error, rather than clear evidence of a substantive numerical failure.

The scenarios with the highest maximum iteration counts all had theta = 0.2, suggesting that smaller-sample regimes may require more solver iterations.

These saved diagnostics do not independently establish whether warnings occurred during the full simulation.

### 5.2 Warning and Exception Check

A separate stability experiment examined 27 selected scenarios covering:

- Three dimensions: p = 128, 256, 512
- Three sparsity regimes
- Three theta values: 0.2, 1.0, 2.4
- Ten replications per selected scenario

A total of 270 additional simulation runs were executed with Python warnings explicitly captured.

| Diagnostic | Count |
|---|---:|
| Total runs | 270 |
| Warnings | 0 |
| Convergence warnings | 0 |
| Exceptions | 0 |

No numerical warnings, convergence warnings, or exceptions were observed in the sampled stability experiment.

This finding applies to the 270 inspected runs and does not establish that warnings were absent from every replication in the original full simulation.

## 6. Identified Bottlenecks and Next Steps

Profiling identifies two primary optimization opportunities:

**Data generation:** The original implementation generates a dense Gaussian design matrix and computes the response through multiplication by a sparse coefficient vector. Exploiting the known support may reduce arithmetic work, although array-copying costs and floating-point differences must be assessed.

**Lasso fitting:** Repeated independent model fits dominate cumulative execution time. Parallel execution across appropriately grouped simulation tasks is a promising candidate, subject to process overhead and reproducibility constraints.

The next phase will evaluate optimizations from at least two categories, measure their performance impact, and validate that the statistical results remain equivalent to the baseline.

All optimizations will be benchmarked against the preserved Unit 2 implementation and documented in `docs/OPTIMIZATION.md`.
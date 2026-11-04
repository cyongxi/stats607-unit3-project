# Optimization Implementation and Performance Analysis

## 1. Overview

This project optimizes the Unit 2 simulation study reproducing the Lasso signed-support recovery experiments from Wainwright (2009), Figure 2.

The original study contains 126 scenarios with 200 Monte Carlo replications each, totaling 25,200 Lasso fits.

Following the workflow introduced in STATS 607 Lecture 08, the optimization process consisted of:

1. Establishing correctness and baseline performance.
2. Profiling the original implementation.
3. Identifying computational bottlenecks.
4. Implementing and benchmarking candidate optimizations.
5. Verifying numerical and statistical correctness.

Two optimization categories were implemented:

- **Algorithmic Improvements:** Elimination of redundant scenario-specific computations.
- **Parallelization:** Parallel execution of independent simulation scenarios.

Additional experiments evaluated sparse-support matrix multiplication and Fortran-contiguous memory layout. Neither was retained because the available full-simulation benchmarks did not demonstrate a meaningful performance improvement.

## 2. Baseline Performance and Bottlenecks

The original serial simulation was benchmarked three times.

| Run | Wall-clock runtime |
|---|---:|
| 1 | 76.97 s |
| 2 | 73.75 s |
| 3 | 73.70 s |

The baseline median runtime was **73.75 seconds**.

Python `cProfile` identified two major computational bottlenecks:

| Function | Calls | Cumulative time |
|---|---:|---:|
| `fit_lasso()` | 25,200 | 49.630 s |
| `generate_data()` | 25,200 | 30.892 s |
| `exact_recovery()` | 25,200 | 0.282 s |
| `calculate_lambda()` | 25,200 | 0.042 s |

Lasso fitting and data generation accounted for approximately 97% of the measured `run_simulation()` time.

These findings motivated the investigation of parallel execution and opportunities to eliminate redundant computations.

Detailed baseline results are documented in `docs/BASELINE.md`.

## 3. Optimization 1: Elimination of Redundant Computations

### 3.1 Problem Identified

In the original implementation, each replication repeatedly extracted and converted scenario parameters.

Additionally, the Lasso regularization parameter was recalculated for every replication, even though its value was constant within each scenario.

The regularization parameter is

\[
\lambda_n =
\sqrt{\frac{2\sigma^2\log(k)\log(p-k)}{n}}.
\]

Because \(n\), \(p\), \(k\), and \(\sigma\) are fixed within a scenario, the value can be calculated once and reused across its 200 replications.

### 3.2 Solution Implemented

The `fit_lasso()` function in `src/methods.py` was extended to accept an optional precomputed `lam` argument.

**Before:**

```python
def fit_lasso(X, y, k, sigma, tol=1e-7, max_iter=20000):
    # Other validation and fitting code omitted.
    lam = calculate_lambda(n, p, k, sigma)
```

**After:**

```python
def fit_lasso(
    X, y, k, sigma,
    tol=1e-7,
    max_iter=20000,
    lam=None,
):
    # Other validation and fitting code omitted.
    if lam is None:
        lam = calculate_lambda(n, p, k, sigma)
```

This preserves compatibility with existing function calls.

In `src/simulation.py`, scenario-specific parameters and the regularization parameter are calculated outside the inner replication loop.

```python
n = int(scenario["n"])
p = int(scenario["p"])
k = int(scenario["k"])
sigma = float(scenario["sigma"])
beta_min = float(scenario["beta_min"])

lam = calculate_lambda(n, p, k, sigma)

for replication_id in range(reps):
    # Generate data.
    # Fit Lasso using lam=lam.
    # Record results.
```

The same lambda precomputation strategy was subsequently applied to the parallel implementation.

### 3.3 Performance Evidence

The original profiler recorded 25,200 calls to `calculate_lambda()`.

After optimization, the profiler recorded only 126 calls.

This represents a 99.5% reduction in the number of regularization-parameter calculations.

The updated serial simulation was also benchmarked:

| Configuration | Runtime |
|---|---:|
| Original serial baseline, three-run median | 73.75 s |
| Precomputed serial, initial single run | 77.78 s |
| Precomputed serial, later three-run median | 75.45 s |

The optimized profiler reported:

| Function | Original cumulative time | Optimized cumulative time |
|---|---:|---:|
| `run_simulation()` | 82.993 s | 81.884 s |
| `fit_lasso()` | 49.630 s | 49.479 s |
| `generate_data()` | 30.892 s | 30.920 s |

Although the redundant calculations were successfully eliminated, the available wall-clock measurements did not demonstrate a meaningful overall speedup.

This is consistent with the original profiling result: `calculate_lambda()` accounted for only approximately 0.042 seconds of the full simulation.

### 3.4 Trade-offs

The optimization reduces redundant work and makes the separation between scenario-level and replication-level computations more explicit.

However, it adds an optional argument to `fit_lasso()` and requires callers to manage the precomputed parameter correctly.

The implementation therefore improves computational organization, but its contribution to total runtime is negligible relative to Lasso fitting and data generation.

This illustrates the importance of using profiling results and Amdahl's Law when evaluating optimization opportunities.

## 4. Optimization 2: Scenario-Level Parallelization

### 4.1 Problem Identified

The original simulation processes all 126 scenarios sequentially.

Each scenario contains independent Monte Carlo replications, making the workload suitable for parallel execution.

The original implementation uses NumPy `SeedSequence` to generate reproducible random streams for individual replications.

Preserving the assignment of these random streams was essential for ensuring that parallel execution did not change the simulation results.

### 4.2 Solution Implemented

A separate implementation was created in `src/parallel.py` using Python's `concurrent.futures.ProcessPoolExecutor`.

Replications were grouped by scenario, and each scenario was assigned a deterministic sequence of child seeds.

The main process prepared the tasks and assigned them to worker processes.

```python
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    batches = list(executor.map(run_scenario, tasks))
```

The use of `executor.map()` preserves the order of returned scenario results.

The implementation supports configurable worker counts through the `--workers` command-line argument.

The original serial implementation was retained to facilitate performance comparisons and correctness validation.

### 4.3 Initial Performance Results

Initial full-simulation runs produced:

| Configuration | Wall-clock runtime |
|---|---:|
| Original serial, median | 73.75 s |
| Parallel, 2 workers | 43.29 s |
| Parallel, 4 workers | 29.99 s |
| Parallel + lambda precomputation, 4 workers | 29.80 s |

The combined implementation initially achieved approximately 2.47× speedup relative to the original serial median.

These were preliminary comparisons. Worker configurations were not yet evaluated under the final repeated benchmark protocol.

### 4.4 Repeated Benchmark

A benchmark script was implemented in `scripts/benchmark.py`.

The benchmark used the same configured BLAS thread limits for each execution mode and measured three repetitions per configuration.

The serial configuration in this benchmark included lambda precomputation. Therefore, this experiment measures the additional benefit of parallelization relative to the updated serial implementation, rather than independently measuring the original unmodified Unit 2 code.

Results:

| Configuration | Median runtime | Mean runtime | Speedup |
|---|---:|---:|---:|
| Precomputed serial | 75.451 s | 77.210 s | 1.00× |
| Parallel, 2 workers | 57.900 s | 53.999 s | 1.30× |
| Parallel, 4 workers | 48.413 s | 44.263 s | 1.56× |

The four-worker implementation reduced median wall-clock runtime by approximately 35.8% relative to the updated serial implementation.

The repeated benchmark results provide stronger evidence of performance improvement than the earlier single-run measurements.

However, substantial runtime variability was observed, especially in the parallel configurations.

Possible contributors include operating-system scheduling, competing background processes, processor thermal behavior, and process-management overhead. These causes were not isolated experimentally.

### 4.5 Trade-offs

Parallel execution substantially reduces wall-clock runtime for this independent Monte Carlo workload.

However, the implementation introduces additional complexity:

- Process creation and coordination overhead.
- Increased concurrent memory usage.
- More complicated random-stream management.
- Potential competition for CPU resources.
- Dependence on worker count and hardware characteristics.

The results also show that parallel scaling is sublinear: doubling the number of workers does not produce a proportional reduction in runtime.

This is consistent with Amdahl's Law and the practical costs of multiprocessing.

## 5. Additional Optimization Experiments

### 5.1 Sparse-Support Matrix Multiplication

The original response calculation was:

```python
y = X @ beta_true + noise
```

An alternative implementation exploited the known sparse support:

```python
y = X[:, support] @ beta_true[support] + noise
```

The multiplication's arithmetic workload decreases from approximately \(O(np)\) to \(O(nk)\), where \(k<p\).

However, NumPy advanced indexing requires additional memory operations and typically creates an intermediate array.

Microbenchmark results:

| Dimensions (n, p, k) | Original | Sparse | Speedup |
|---|---:|---:|---:|
| (200, 128, 52) | 0.0015 ms | 0.0103 ms | 0.147× |
| (500, 256, 103) | 0.0035 ms | 0.0531 ms | 0.067× |
| (1000, 512, 205) | 0.0122 ms | 0.2351 ms | 0.052× |

The sparse-support implementation was slower in all three tested configurations.

A full simulation run using this modification took 79.88 seconds, compared with the original baseline median of 73.75 seconds. This single full-simulation measurement was not sufficient to establish a stable runtime difference, but the microbenchmark consistently demonstrated unfavorable performance.

Full-result validation found identical recovery outcomes across all 25,200 replications.

The optimization was therefore rejected, and the original matrix multiplication was restored.

### 5.2 Fortran-Contiguous Memory Layout

A second experiment investigated whether column-major storage would improve scikit-learn's coordinate-descent fitting performance.

The candidate modification used:

```python
X = np.asarray(X, dtype=float, order="F")
```

End-to-end microbenchmarks, including the conversion cost, produced speedups between approximately 1.03× and 1.07× across three tested dimensions.

Numerical comparisons showed identical fitted coefficients in those tests.

However, the full serial simulation using this memory-layout modification took 76.13 seconds, compared with the original baseline median of 73.75 seconds.

Because the complete workload did not demonstrate a convincing performance benefit, this modification was not retained.

These experiments illustrate that local performance improvements do not necessarily translate into whole-program speedups.

## 6. Regression Tests and Correctness

The project includes 15 automated tests covering data generation, Lasso estimation, support recovery, reproducibility, and serial-parallel equivalence.

The complete test suite was executed using:

```bash
make test
```

All 15 tests passed.

Additional full-simulation validations compared the original serial results with the optimized versions across all 25,200 replications.

The comparisons included:

- Scenario and replication identifiers.
- Signed-support recovery outcomes.
- Regularization parameters.
- Lasso iteration counts.
- Reported duality gaps.

The comparisons passed using numerical tolerances of `rtol=1e-10` and `atol=1e-10`.

These validations support the conclusion that the implemented optimizations preserved the recorded statistical results for the tested simulation design.

## 7. Numerical Stability

A separate stability-check script examined 27 selected scenarios, with 10 replications per scenario, for a total of 270 runs.

The optimized version produced:

| Diagnostic | Count |
|---|---:|
| Total runs | 270 |
| Warnings | 0 |
| Convergence warnings | 0 |
| Exceptions | 0 |
| Runs reaching max_iter | 0 |
| Non-finite dual gaps | 0 |

No numerical or convergence problems were observed in this sampled check.

The results do not establish that warnings are impossible in other simulation settings.

## 8. Reproducibility and Automation

The updated Makefile provides the following targets:

| Target | Purpose |
|---|---|
| `make profile` | Profile the updated serial simulation |
| `make complexity` | Analyze empirical computational scaling |
| `make benchmark` | Benchmark serial and parallel configurations |
| `make parallel` | Run the four-worker parallel simulation |
| `make stability-check` | Check numerical warnings and convergence diagnostics |

The benchmark script saves raw timing data, summary statistics, and a performance comparison figure.

The benchmark visualization is saved as:

`results/figures/optimization_runtime_comparison.png`

The performance measurements should be interpreted in the context of the reported execution environment and experimental design.

## 9. Lessons Learned

The most important improvement came from parallelizing independent Monte Carlo scenarios.

The original profiling results showed that the simulation spent most of its time in repeated Lasso fitting and data generation. Running independent scenarios concurrently reduced overall wall-clock time without changing the statistical results.

Precomputing lambda successfully eliminated nearly all redundant regularization-parameter calculations but had little observable effect on total runtime. This was expected because the original operation accounted for a very small fraction of the simulation time.

The sparse-support experiment was particularly informative. Although it reduced the theoretical number of arithmetic operations, the additional memory-access costs made it slower in practice.

Similarly, Fortran-contiguous storage improved some isolated Lasso fitting benchmarks but did not produce a convincing full-simulation improvement.

Overall, this project demonstrated that computational complexity, profiler measurements, and end-to-end wall-clock performance provide different but complementary information.

The most effective optimization strategy is to identify dominant bottlenecks, optimize them selectively, benchmark representative workloads, and verify correctness after every change.
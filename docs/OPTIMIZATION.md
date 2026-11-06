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

### 4.4 Controlled Baseline-versus-Optimized Benchmark

A controlled benchmark was conducted to compare the original Unit 2 implementation with the optimized Unit 3 implementations.

The original implementation was recovered from the `unit3-baseline` Git tag (commit `ededf3b`) and executed from an independent Git worktree.

All configurations used the same Python 3.11.1 virtual environment, dependency versions, simulation design, and thread limits:

- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

Each configuration completed 126 simulation scenarios with 200 replications per scenario, totaling 25,200 Lasso fits.

Each configuration was benchmarked three times using `/usr/bin/time -p`. The primary performance metric was median wall-clock time (`real`).

Five configurations were evaluated:

1. **Original:** Unmodified serial implementation.
2. **Algorithmic-only:** Serial implementation with scenario-specific parameter precomputation.
3. **Parallel-only (4 workers):** Parallel implementation without lambda precomputation, retaining some scenario-level parameter caching.
4. **Combined (2 workers):** Parallel execution with lambda precomputation, using two workers.
5. **Combined (4 workers):** Parallel execution with lambda precomputation, using four workers.

#### Controlled Benchmark Results

| Configuration | Median Real (s) | Mean Real (s) | Median CPU (s) | Speedup | Runtime Reduction |
|---|---:|---:|---:|---:|---:|
| Original | 75.07 | 75.567 | 74.78 | 1.000x | 0.0% |
| Algorithmic-only | 76.83 | 76.363 | 75.80 | 0.977x | -2.3% |
| Parallel-only (4) | 30.50 | 30.477 | 115.86 | 2.461x | 59.4% |
| Combined (2) | 45.10 | 45.260 | 88.25 | 1.665x | 39.9% |
| Combined (4) | 31.87 | 31.980 | 121.35 | 2.356x | 57.5% |

Speedup is defined as:

\[
S_p = \frac{T_{\mathrm{original}}}{T_p},
\]

where the original serial median runtime is 75.07 seconds.

The combined four-worker implementation achieved a 2.356x speedup, reducing median wall-clock runtime by approximately 57.5%.

Algorithmic-only execution was 2.3% slower in observed median wall-clock time. This difference does not establish a systematic slowdown or a meaningful speedup.

The parallel-only four-worker configuration achieved a lower observed median runtime than the corrected combined four-worker configuration (30.50 versus 31.87 seconds). However, the difference is small relative to run-to-run variability and does not establish a reliable additional speedup from lambda precomputation.

The experiments were conducted sequentially rather than fully randomized. Consequently, temperature, operating-system scheduling, and background activity may have influenced the measurements.

The parallel-only configuration specifically removes lambda precomputation while retaining some scenario-level parameter caching; it should not be interpreted as a complete isolation of every algorithmic modification.

#### Performance Visualizations

The controlled runtime comparison is available at:

`results/figures/controlled_runtime_comparison.png`

The CPU resource trade-off comparison is available at:

`results/figures/controlled_cpu_tradeoff.png`

The runtime figure reports the median of three measurements for each configuration, with error bars indicating the observed minimum-to-maximum range rather than confidence intervals.

Raw measurements and summary statistics are saved in:

- `results/benchmarks/controlled_benchmark_raw.csv`
- `results/benchmarks/controlled_benchmark_summary.csv`

Earlier exploratory benchmarks are retained as historical evidence but are not used as the primary estimates of the final controlled speedup.

### 4.5 Performance and Resource Trade-offs

Parallelization substantially reduces elapsed time but increases aggregate CPU time.

The `/usr/bin/time -p` measurements distinguish:

- `real`: Elapsed wall-clock time.
- `user`: Aggregate CPU time spent executing user-space code.
- `sys`: Aggregate CPU time spent executing kernel-space operations.

Aggregate CPU time is calculated as `user + sys` for each run, followed by the median across three runs.

| Configuration | Median Wall Time (s) | Median CPU Time (s) |
|---|---:|---:|
| Original | 75.07 | 74.78 |
| Combined (2 workers) | 45.10 | 88.25 |
| Combined (4 workers) | 31.87 | 121.35 |

Relative to the original serial implementation:

- Two workers reduced median wall-clock time by 39.9%, while increasing aggregate CPU time by approximately 18.0%.
- Four workers reduced median wall-clock time by 57.5%, while increasing aggregate CPU time by approximately 62.3%.

Parallel efficiency is defined as:

\[
E_p = \frac{T_1}{pT_p},
\]

where \(p\) is the number of worker processes.

Using the original serial median as \(T_1\), the approximate parallel efficiencies were:

| Configuration | Speedup | Parallel Efficiency |
|---|---:|---:|
| Combined (2 workers) | 1.665x | 83.2% |
| Combined (4 workers) | 2.356x | 58.9% |

Among the combined configurations, four workers minimized completion time; the parallel-only four-worker configuration had the lowest observed median runtime overall.

The two-worker configuration provides a better trade-off when aggregate CPU consumption and resource efficiency are important.

The additional CPU cost of multiprocessing may reflect process management, repeated initialization, memory operations, scheduling, and resource contention. These factors were not isolated individually.

Therefore, this project demonstrates a latency-oriented optimization rather than a reduction in total computational work.

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

The project uses a Makefile to automate simulation, profiling, benchmarking, testing, and numerical stability checks.

### 8.1 Makefile Targets

| Target | Purpose |
|---|---|
| `make simulate` | Run the updated serial simulation |
| `make parallel` | Run the combined four-worker simulation |
| `make profile` | Profile the updated serial implementation |
| `make complexity` | Analyze empirical computational scaling |
| `make benchmark` | Re-run the controlled baseline-versus-optimized benchmark |
| `make benchmark-report` | Regenerate benchmark summaries and figures from existing measurements |
| `make stability-check` | Examine warnings, convergence, and numerical stability |
| `make test` | Run automated correctness and regression tests |

### 8.2 Reproducing the Original Baseline

The original unoptimized implementation is preserved in the Git tag `unit3-baseline` (commit `ededf3b`).

To reproduce the controlled comparison, create an independent Git worktree alongside the main project directory:

```bash
git worktree add --detach ../stats607-unit3-baseline unit3-baseline
```

The benchmark script checks that this worktree points to the expected baseline commit.

Both the baseline and optimized implementations use the same Python interpreter from the main project's virtual environment.

The following thread limits are applied during controlled measurements:

```bash
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

The current benchmark automation assumes a macOS environment with `/usr/bin/time -p` available.

### 8.3 Reproducing the Benchmarks

To rerun all five configurations with three repetitions each:

```bash
make benchmark
```

The five configurations include the original serial version, algorithmic-only serial version, parallel-only four-worker version, and combined versions with two and four workers.

The parallel-only configuration is an additional experimental implementation in `src/parallel_only.py`.

To regenerate reports and visualizations without rerunning the simulations:

```bash
make benchmark-report
```

This requires the existing `controlled_benchmark_raw.csv` file.

### 8.4 Benchmark Outputs

The controlled benchmark produces:

| File | Description |
|---|---|
| `results/benchmarks/controlled_benchmark_raw.csv` | Individual timing measurements |
| `results/benchmarks/controlled_benchmark_summary.csv` | Runtime summaries, CPU consumption, and speedup |
| `results/figures/controlled_runtime_comparison.png` | Original-versus-optimized performance comparison |
| `results/figures/controlled_cpu_tradeoff.png` | Wall-clock runtime versus aggregate CPU time |

Raw simulation CSV files and binary profiler outputs are excluded from Git because they can be regenerated.

Earlier exploratory benchmark results are retained separately.

### 8.5 Reproducibility Considerations

All simulations use the same experimental design and master random seed.

Full-result regression validation confirmed numerical agreement between the original and optimized implementations across 25,200 simulation records within the specified tolerance.

Wall-clock benchmark measurements may vary across systems and executions because of hardware differences, background workloads, operating-system scheduling, and resource contention.

Reported performance improvements therefore apply to the tested computational environment rather than representing universal speedups.

## 9. Lessons Learned

This project demonstrated that optimizing a statistical simulation requires distinguishing between algorithmic efficiency, execution time, and computational resource consumption.

### 9.1 Profiling Before Optimization

The original profiling results identified Lasso fitting and data generation as the dominant computational bottlenecks.

These results helped prioritize optimization strategies and avoid spending excessive effort on operations with negligible contributions to total runtime.

### 9.2 Algorithmic Improvements Do Not Always Produce Significant Speedups

Precomputing the Lasso regularization parameter reduced the number of calls to `calculate_lambda()` from 25,200 to 126.

However, the controlled serial benchmark showed almost no change in overall runtime:

- Original serial median: 75.07 seconds.
- Algorithmic-only median: 76.83 seconds.

The observed difference was too small to establish a meaningful improvement.

This result illustrates Amdahl's Law: optimizing a small fraction of total execution time has limited impact on overall performance.

### 9.3 Parallelization Was the Main Source of Performance Improvement

Scenario-level parallelization allowed independent Monte Carlo simulations to execute concurrently.

The combined four-worker implementation reduced median wall-clock runtime from 75.07 to 31.87 seconds.

This represents approximately 2.36x speedup and a 57.5% reduction in elapsed time.

The optimization preserved the statistical simulation design and produced results consistent with the original implementation.

### 9.4 Faster Execution Can Require More CPU Resources

The combined four-worker implementation increased median aggregate CPU time from 74.78 to 121.35 seconds.

Two workers provided a smaller speedup but substantially better parallel efficiency.

Consequently, the preferred worker count depends on whether the objective is to minimize elapsed time or computational resource consumption.

### 9.5 Microbenchmarks Must Be Validated End-to-End

Experiments involving sparse-support matrix multiplication and Fortran-contiguous array storage demonstrated that theoretically attractive or locally faster operations do not necessarily improve the complete simulation.

The final implementation retained only modifications supported by correctness checks and meaningful performance evidence.

### 9.6 Overall Conclusion

The principal contribution of this optimization project is a reproducible, parallel simulation workflow that substantially reduces time-to-completion while preserving the statistical results.

The experiments emphasize the importance of profiling, controlled benchmarking, regression testing, and explicit evaluation of performance trade-offs.
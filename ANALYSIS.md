# Analysis — Reproduction of Wainwright (2009), Figure 2

## 1. Replication Results

I reproduced Figure 2 of Wainwright (2009), which studies exact signed-support recovery by the Lasso under an independent Gaussian design. The simulation covers three sparsity regimes (linear, sublinear, and fractional-power), $p\in\{128,256,512\}$, and 14 values of the rescaled sample size $\theta$. I ran 200 replications for each of 126 scenarios, giving 25,200 Lasso fits.

The results reproduce the main pattern: recovery probability is low at small $\theta$, rises rapidly around the threshold region, and approaches one at larger sample sizes. For $p=512$, recovery probability increases from 0.125 to 0.985 under linear sparsity as $\theta$ increases from 0.8 to 2.0. The corresponding increases are 0.070 to 0.965 for sublinear sparsity and 0.085 to 0.955 for fractional-power sparsity.

I also estimated $\theta_{50}$, the point where the estimated recovery probability first reaches 0.5, using linear interpolation between neighboring simulation points:

| Sparsity regime | $p=128$ | $p=256$ | $p=512$ |
|---|---:|---:|---:|
| Linear | 1.068 | 1.022 | 1.000 |
| Sublinear | 1.250 | 1.165 | 1.149 |
| Fractional-power | 1.206 | 1.145 | 1.112 |

In each regime, $\theta_{50}$ moves closer to one as $p$ increases. This agrees with the direction suggested by the theoretical threshold, although three finite dimensions cannot establish asymptotic convergence. The interpolated estimates also have Monte Carlo uncertainty that I have not separately quantified.

## 2. Comparison with the Original Figure

My recovery curves reproduce the S-shaped pattern in the original Figure 2, but the transitions generally occur at larger values of $\theta$, especially under sublinear and fractional-power sparsity. At $\theta=1$ and $p=512$, my recovery probabilities are 0.500, 0.295, and 0.325 across the three regimes. The corresponding points appear higher in the published figure. Since the original numerical data are unavailable, this comparison remains visual.

I checked the data-generating model and Lasso objective against the paper. The penalty follows

$$
\lambda_n=\sqrt{\frac{2\sigma^2\log(k)\log(p-k)}{n}},
$$

and scikit-learn uses the same $1/(2n)$ loss scaling as Equation (3).

One complication is that Section VII specifies $n=2\theta k\log(p-k)$, while the Figure 2 caption defines $\theta=n/[k\log(p-k)]$. I followed Section VII:

$$
\theta=\frac{n}{2k\log(p-k)},\qquad
n=\left\lceil2\theta k\log(p-k)\right\rceil.
$$

Using the caption's alternative definition would double my horizontal coordinates rather than resolve the rightward shift. Other possible sources of disagreement include the original simulation grid, solver settings, random seeds, and classification of near-zero coefficients. I cannot determine their individual effects from the available information.

## 3. Critique of the Simulation Design

The original design is closely tied to the theory. By varying sparsity and dimension under a controlled Gaussian model, it isolates the signed-support recovery threshold.

However, Figure 2 evaluates only the Lasso, so it does not establish how Lasso compares with alternative selection methods. Independent Gaussian predictors and coefficients fixed at $\pm0.5$ also create an idealized setting. These choices are useful for studying the threshold, but may not represent practical datasets with correlated predictors, varying signal strengths, or non-Gaussian noise. The paper studies correlated designs elsewhere, but my replication focuses on Figure 2.

I would extend the study by varying signal strength and predictor correlation, comparing Lasso with another selection method on the same datasets, and using a denser $\theta$ grid near the transition. I would also test sensitivity to the numerical zero threshold. These extensions would help assess whether the observed pattern is robust beyond the original conditions.

## 4. Visualizations and Implementation

I recreated the three-panel recovery plot (`results/figures/recovery_curves.png`) and added error bars of $1.96\times\widehat{\mathrm{MCSE}}$. With 200 replications per scenario, the maximum plug-in MCSE is approximately 0.035. These error bars help distinguish broad patterns from small Monte Carlo fluctuations, although normal-approximation intervals are less reliable near probabilities of zero or one.

I also created a solver-iteration diagnostic plot (`results/figures/solver_diagnostics.png`). The maximum iteration count was 106, far below the limit of 20,000, and the maximum recorded duality gap was approximately $3.43\times10^{-6}$. These results are reassuring, although they do not guarantee exact numerical solutions.

The main implementation challenge was translating the paper's mathematical definitions into consistent simulation code and investigating the differences from the original curves. I separated data generation, estimation, and evaluation into modules and checked reproducibility using a fixed seed. All 27 automated tests passed, and the full results contain 25,200 records with no missing values.

I am confident in the implementation and the overall recovery pattern, but less confident in exact numerical replication because some original computational settings remain unspecified.

## 5. Conclusion

The replication captures the main qualitative finding of Wainwright (2009): exact signed-support recovery increases rapidly as the rescaled sample size grows. However, the transition locations do not match the original figure exactly. I retained the documented implementation rather than adjusting parameters solely to match the published curves, and report the remaining differences as a limitation.

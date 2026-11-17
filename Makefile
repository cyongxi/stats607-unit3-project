# Override when needed:
# make PYTHON=.venv-1/bin/python all
PYTHON := .venv/bin/python

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

RAW := results/raw/simulation_results.csv
SUMMARY := results/recovery_summary.csv
RECOVERY_FIG := results/figures/recovery_curves.png
DIAGNOSTIC_FIG := results/figures/solver_diagnostics.png

# Changes to these files require rerunning the simulation.
SIMULATION_SOURCES := \
	src/__init__.py \
	src/simulation.py \
	src/scenarios.py \
	src/dgps.py \
	src/methods.py \
	src/metrics.py

.PHONY: all simulate analyze figures test clean pilot \
	profile complexity benchmark benchmark-report parallel \
	stability-check regression-baseline

# Public commands delegate to actual output files.
all: figures

simulate: $(RAW)

analyze: $(SUMMARY)

figures: $(RECOVERY_FIG) $(DIAGNOSTIC_FIG)


# ----- Statistical simulation pipeline -----

$(RAW): $(SIMULATION_SOURCES) requirements.txt
	$(PYTHON) -m src.simulation

$(SUMMARY): $(RAW) src/analyze.py src/__init__.py requirements.txt
	$(PYTHON) -m src.analyze

# src.figures generates both figures in one invocation.
# Use recovery_curves.png as the primary output to serialize the work.
$(RECOVERY_FIG): $(SUMMARY) $(RAW) src/figures.py src/__init__.py requirements.txt
	$(PYTHON) -m src.figures
	test -s "$(RECOVERY_FIG)" && test -s "$(DIAGNOSTIC_FIG)"

# Restore the second figure if it was deleted independently.
# Depending on the primary output prevents simultaneous plotting under make -j.
$(DIAGNOSTIC_FIG): $(RECOVERY_FIG)
	@if test ! -s "$(DIAGNOSTIC_FIG)" || \
	   test "$(RECOVERY_FIG)" -nt "$(DIAGNOSTIC_FIG)"; then \
		$(PYTHON) -m src.figures; \
	fi
	@test -s "$(DIAGNOSTIC_FIG)"


# ----- Correctness and regression validation -----

test:
	$(PYTHON) -m pytest tests/ -v

regression-baseline:
	$(PYTHON) -m scripts.validate_baseline


# ----- Explicit simulation runs -----

pilot:
	$(PYTHON) -m src.simulation --pilot

parallel:
	$(PYTHON) -m src.parallel --workers 4


# ----- Profiling and performance analysis -----

profile:
	mkdir -p results/profiling
	$(PYTHON) -m cProfile -o results/profiling/optimized_full.prof -m src.simulation
	$(PYTHON) -c "import pstats; pstats.Stats('results/profiling/optimized_full.prof').strip_dirs().sort_stats('cumulative').print_stats(20)"

complexity:
	$(PYTHON) -m scripts.complexity

benchmark:
	$(PYTHON) -m scripts.benchmark

# Reporting uses saved measurements; it does not trigger a new benchmark.
benchmark-report: results/benchmarks/controlled_benchmark_raw.csv
	$(PYTHON) -m scripts.benchmark --report-only

stability-check:
	$(PYTHON) -m scripts.stability_check


# ----- Remove reproducible simulation outputs -----

# Preserve recorded benchmark measurements and performance figures.
clean:
	rm -f "$(RAW)"
	rm -f "$(SUMMARY)"
	rm -f "$(RECOVERY_FIG)"
	rm -f "$(DIAGNOSTIC_FIG)"
	rm -f results/raw/pilot_results.csv
	rm -f results/raw/parallel_results.csv
	rm -f results/raw/parallel_pilot.csv
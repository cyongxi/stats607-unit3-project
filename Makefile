PYTHON := .venv/bin/python

.PHONY: all simulate analyze figures test clean pilot profile complexity benchmark parallel stability-check

all: simulate analyze figures

simulate:
	$(PYTHON) -m src.simulation

analyze:
	$(PYTHON) -m src.analyze

figures:
	$(PYTHON) -m src.figures

test:
	$(PYTHON) -m pytest tests/ -v

pilot:
	$(PYTHON) -m src.simulation --pilot

parallel:
	$(PYTHON) -m src.parallel --workers 4

complexity:
	$(PYTHON) -m scripts.complexity

clean:
	rm -f results/raw/simulation_results.csv
	rm -f results/recovery_summary.csv
	rm -f results/figures/*.png

benchmark:
	$(PYTHON) -m scripts.benchmark

profile:
	mkdir -p results/profiling
	$(PYTHON) -m cProfile -o results/profiling/optimized_full.prof -m src.simulation
	$(PYTHON) -c "import pstats; pstats.Stats('results/profiling/optimized_full.prof').strip_dirs().sort_stats('cumulative').print_stats(20)"

stability-check:
	$(PYTHON) -m scripts.stability_check
# Phase 1. `make figures` on a clean clone runs every benchmark it needs.
# naive is serial and slow: NAIVE_N requests, NAIVE_C concurrency points only.
PY      ?= uv run python
MAXC    ?= 64
CONC    ?= 1,2,4,8,16,32,64
NAIVE_N ?= 32
NAIVE_C ?= 1,4,16,64
RES     := bench/results
FIG     := figures/out

.PHONY: install test serve trace seqlen sweep sweep-naive sweep-static sweep-continuous figures hardware clean

install:
	uv sync --group dev

test:
	uv run pytest

serve:
	uv run uvicorn engine.server:app --host 0.0.0.0 --port 8000

trace: bench/traces/mixed.json
bench/traces/mixed.json: bench/make_trace.py
	$(PY) -m bench.make_trace

hardware:
	$(PY) -m bench.hardware

# --- benchmarks ------------------------------------------------------------

seqlen: $(RES)/seqlen_naive.json $(RES)/seqlen_cached.json
$(RES)/seqlen_naive.json:
	$(PY) -m bench.harness seqlen --mode naive
$(RES)/seqlen_cached.json:
	$(PY) -m bench.harness seqlen --mode cached

sweep-naive: bench/traces/mixed.json
	$(PY) -m bench.harness sweep --mode naive --concurrency $(NAIVE_C) --num-requests $(NAIVE_N)
sweep-static: bench/traces/mixed.json
	$(PY) -m bench.harness sweep --mode static --concurrency $(CONC)
sweep-continuous: bench/traces/mixed.json
	$(PY) -m bench.harness sweep --mode continuous --concurrency $(CONC)
sweep: sweep-naive sweep-static sweep-continuous

# The three headline runs, same command three times.
bench-naive bench-static bench-continuous: bench-%: bench/traces/mixed.json
	$(PY) -m bench.harness trace --mode $* --concurrency $(MAXC) $(if $(filter naive,$*),--num-requests $(NAIVE_N))

# --- figures ---------------------------------------------------------------

$(RES)/.sweep-stamp:
	$(MAKE) sweep
	touch $@

figures: seqlen $(RES)/.sweep-stamp
	$(PY) -m figures.fig_naive_vs_seqlen
	$(PY) -m figures.fig_batch_occupancy --concurrency $(MAXC)
	$(PY) -m figures.fig_throughput_sweep

figures-only:
	$(PY) -m figures.fig_naive_vs_seqlen
	$(PY) -m figures.fig_batch_occupancy
	$(PY) -m figures.fig_throughput_sweep

clean:
	rm -rf $(RES)/*.json $(RES)/.sweep-stamp $(FIG)

PYTHON ?= .venv/bin/python
CONFIG ?= config.yaml
API_URL ?= http://localhost:8000
STAGES ?= load,clean,split,retrieval,features,train,evaluate,export

.PHONY: install train test lint check api bench app clean-models

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-deps

train:
	$(PYTHON) -m shoprank.pipeline --config $(CONFIG) --stages $(STAGES)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

check: lint test

api:
	@test -f api/main.py || { echo "api/main.py does not exist yet (Phase 10)"; exit 1; }
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000

bench:
	$(PYTHON) scripts/benchmark_api.py --url http://localhost:8000 --n 100

app:
	@curl -sf $(API_URL)/health > /dev/null || echo "warning: API not reachable at $(API_URL); start it with make api"
	API_URL=$(API_URL) $(PYTHON) -m streamlit run app/Home.py

clean-models:
	rm -rf models/*.joblib models/*.npy models/*.parquet models/*.json

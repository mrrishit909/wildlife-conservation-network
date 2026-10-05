export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55206/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8206

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m wl.seed --if-empty
dev:            ## API with reload on :8206, jobs run by a local worker
	($(PY) -m core.jobs wl.api &) && $(PY) -m uvicorn wl.api:app --reload --port 8206
test:
	$(PY) -m pytest --cov=wl --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) reserve-researcher-demo 32 10 "GET /v1/alerts/suspicious" "GET /v1/risk/tiles" "GET /v1/corridor/shift" "GET /v1/animals" 
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test reset

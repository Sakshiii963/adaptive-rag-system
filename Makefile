.PHONY: test lint frontend-build compose-config evaluate

test:
	python -m pytest -q

lint:
	ruff check backend evaluation

frontend-build:
	cd frontend && npm run build

compose-config:
	docker compose -f docker/docker-compose.yml config

evaluate:
	python evaluation/run_evaluation.py --workers 2

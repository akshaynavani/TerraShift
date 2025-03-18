.PHONY: install data train evaluate api seed dashboard test up down

PY ?= python

install:
	$(PY) -m pip install -e ".[api,dev]"

data:
	$(PY) -m scripts.generate_synthetic --out data/synthetic --n-per-class 200
	$(PY) -m scripts.generate_synthetic --out data/test --n-per-class 50 --seed 123

train:
	$(PY) -m scripts.train --data data/synthetic --model both --epochs 15

evaluate:
	$(PY) -m scripts.evaluate --data data/test

api:
	uvicorn api.main:app --reload --port 8000

seed:
	$(PY) -m scripts.seed_db --predict

dashboard:
	cd dashboard && npm install && npm run dev

test:
	$(PY) -m pytest -q

up:
	docker compose up --build -d

down:
	docker compose down

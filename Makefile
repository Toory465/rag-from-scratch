.PHONY: install test db l1 l2 l3 l4 l5 l6 l7
install:
	pip install -e ".[pg,dev]"
test:
	pytest -q
db:
	docker compose up -d pgvector
l1:
	python -m lessons.l01_embeddings
l2:
	python -m lessons.l02_chunking
l3:
	python -m lessons.l03_vector_stores
l4:
	python -m lessons.l04_retrieval
l5:
	python -m lessons.l05_reranking
l6:
	python -m lessons.l06_generation
l7:
	python -m lessons.l07_evaluation

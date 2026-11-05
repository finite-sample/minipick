PYTHON ?= python3
.PHONY: check note
check:
	ruff check .
	ruff format --check .
	$(PYTHON) -m pytest -q
note: paper/note.tex paper/references.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/note.tex

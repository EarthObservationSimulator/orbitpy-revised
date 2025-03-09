.PHONY: install test lint format docs

install:
	pip install -e .

test:
	python -m unittest discover -s tests

lint:
	pylint --rcfile=pylintrc orbitpy tests examples # Use the pylintrc file from the Google Python Style Guide 

format:
	black orbitpy tests examples

docs:
	sphinx-build -b html docs/source docs/build
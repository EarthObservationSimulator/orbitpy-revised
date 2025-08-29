.PHONY: install test lint format docs coverage

install:
	pip install -e .

test:
	coverage run -m unittest discover -s tests

slowtest:
	coverage run -m unittest discover -s slowtests

lint:
	pylint --rcfile=pylintrc orbitpy tests examples # Use the pylintrc file from the Google Python Style Guide 

format:
	black orbitpy tests examples

docs:
	sphinx-build -b html docs/source docs/build

coverage:
	coverage report -m
	coverage html
# Style Guide

## References

- [The Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP-8](https://peps.python.org/pep-0008/)
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/)

Take cues from the above references, then use linters, auto-formatters, and LLMs to enforce the styles.

## Notes

- Run `pylint` with the `pylintrc` file (from the Google Python Style Guide) in the root directory:
  ```sh
  pylint --rcfile=pylintrc orbitpy tests examples
  ```

- Run `black` to format the python files.
  ```sh
  black orbitpy tests examples
  ```
- Run `coverage`. The HTML report will be generated in the htmlcov directory. Open htmlcov/index.html in a web browser to view the detailed coverage report.
    ```sh
    coverage run -m unittest discover -s tests
    coverage html
    ```

- Use Google style docstrings.  https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html. The following configuration option in the `conf.py` enables Sphinx to parse Google style dosctrings:  `napoleon_google_docstring = True`

- All OrbitPy objects (classes) intended for use by external users must support initialization and serialization using Python dictionaries. Each class should provide `from_dict(...)` and `to_dict(...)` methods to allow instantiation from dictionary inputs and export their state or data back into a dictionary format.

- All enum values should be in capital letters and formatted with underscores, e.g., `GREGORIAN_DATE`.

- Use only the `from_dict(...)` and other functions that do not depend on external libraries to initialize the OrbitPy objects, even in test functions. External library members should be kept internal to the implementation, as they may change.
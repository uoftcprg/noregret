# noregret

No-regret learning dynamics

## Tests

Run style checks.

```console
flake8 examples noregret
```

Run doctests.

```console
shopt -s globstar
python -m doctest noregret/**/*.py
```

Run unit tests.

```console
python -m unittest
```

Check coverage.

```console
shopt -s globstar
coverage run -m doctest noregret/**/*.py
coverage run -a -m unittest
coverage report -m
coverage html
```

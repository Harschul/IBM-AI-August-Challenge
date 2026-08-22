"""Makes `pytest` (without `python -m`) resolve `src.*` imports.

pytest prepends the directory holding the topmost conftest.py to sys.path, so
this file's existence is what lets `from src.models.contact import Contact`
work from a bare `pytest` invocation -- the form CI and reviewers use. Running
`python -m pytest` happens to work without it because that form puts the
current directory on sys.path itself; nothing else does.

Intentionally empty apart from this note.
"""

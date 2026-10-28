from __future__ import annotations

import ast
import inspect
from pathlib import Path

import picklerick.picklerick_py_native as native

STUB = Path(__file__).resolve().parents[1] / "src" / "picklerick" / "picklerick_py_native.pyi"


def _stub_docs() -> dict[str, str | None]:
    """Docstring of every public name in the stub, keyed "Class.member"."""
    docs: dict[str, str | None] = {}
    for node in ast.parse(STUB.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            docs[node.name] = ast.get_docstring(node)
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, ast.FunctionDef) and not member.name.startswith("_"):
                        docs[f"{node.name}.{member.name}"] = ast.get_docstring(member)
    return docs


def _runtime(name: str) -> object:
    obj: object = native
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def test_stub_declares_the_native_module() -> None:
    stub = {name for name in _stub_docs() if "." not in name}
    runtime = {name for name in dir(native) if not name.startswith("_")}
    assert stub == runtime


def test_stub_docstrings_match_the_rust_doc_comments() -> None:
    for name, doc in _stub_docs().items():
        if doc is None:
            continue
        runtime = inspect.cleandoc(_runtime(name).__doc__ or "")
        assert doc == runtime, f"{name}: stub docstring differs from rust/src/lib.rs"

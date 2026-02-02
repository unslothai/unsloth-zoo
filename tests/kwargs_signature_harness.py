#!/usr/bin/env python
import ast
import json
import os
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.compiler import create_new_function


def _get_func_node(module_ast, name):
    for node in module_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _has_kwargs_alias(module_path, func_name):
    content = Path(module_path).read_text()
    module_ast = ast.parse(content)
    func = _get_func_node(module_ast, func_name)
    if func is None or not func.body:
        return False, False

    has_docstring = (
        isinstance(func.body[0], ast.Expr)
        and isinstance(getattr(func.body[0], "value", None), ast.Constant)
        and isinstance(func.body[0].value.value, str)
    )
    alias_index = 1 if has_docstring else 0
    has_alias = False
    alias_after_docstring = False
    if alias_index < len(func.body):
        stmt = func.body[alias_index]
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                if stmt.targets[0].id == "kwargs":
                    has_alias = True
                    alias_after_docstring = has_docstring
    return has_alias, alias_after_docstring


def run_case(case):
    name = case["name"]
    src = case["src"]
    args = case.get("args", ())
    kwargs = case.get("kwargs", {})
    expect_alias = case.get("expect_alias", False)
    expect_docstring_alias = case.get("expect_docstring_alias", False)
    prepend = case.get("prepend", "")
    append = case.get("append", "")
    is_async = case.get("is_async", False)

    mod = create_new_function(
        name,
        src,
        model_location="math",
        functions=[],
        prepend=prepend,
        append=append,
        overwrite=True,
        add_torch_compile=False,
    )
    func = getattr(mod, name)
    if is_async:
        import asyncio
        out = asyncio.run(func(*args, **kwargs))
    else:
        out = func(*args, **kwargs)

    module_file = getattr(mod, "__file__", None)
    has_alias = False
    alias_after_docstring = False
    if module_file:
        has_alias, alias_after_docstring = _has_kwargs_alias(module_file, name)

    return {
        "name": name,
        "status": "ok",
        "output": out,
        "expect_alias": expect_alias,
        "has_alias": has_alias,
        "alias_ok": has_alias == expect_alias,
        "expect_docstring_alias": expect_docstring_alias,
        "alias_after_docstring": alias_after_docstring,
        "docstring_alias_ok": (not expect_docstring_alias) or alias_after_docstring,
        "module_file": module_file,
    }


def main():
    cases = [
        {
            "name": "alias_positional",
            "src": "def alias_positional(a, kwargs=None, **kwargs):\n    return kwargs if kwargs is not None else a",
            "args": (1, None),
            "expect_alias": True,
        },
        {
            "name": "alias_kwonly",
            "src": "def alias_kwonly(*, kwargs=None, **kwargs):\n    return kwargs",
            "kwargs": {"kwargs": None},
            "expect_alias": True,
        },
        {
            "name": "alias_annotated",
            "src": "def alias_annotated(kwargs: dict = None, **kwargs):\n    return kwargs",
            "args": (None,),
            "expect_alias": True,
        },
        {
            "name": "alias_posonly",
            "src": "def alias_posonly(kwargs, /, x=1, **kwargs):\n    return kwargs",
            "args": (None,),
            "expect_alias": True,
        },
        {
            "name": "alias_varargs",
            "src": "def alias_varargs(*args, kwargs=None, **kwargs):\n    return kwargs",
            "kwargs": {"kwargs": None},
            "expect_alias": True,
        },
        {
            "name": "alias_lambda",
            "src": "def alias_lambda(kwargs=lambda x, y: x + y, **kwargs):\n    return kwargs(1, 2)",
            "args": (),
            "expect_alias": True,
        },
        {
            "name": "alias_fstring",
            "src": "def alias_fstring(kwargs=f\"{1,2}\", **kwargs):\n    return kwargs",
            "args": (),
            "expect_alias": True,
        },
        {
            "name": "alias_str_paren",
            "src": "def alias_str_paren(kwargs=\"a, (b, c)\", **kwargs):\n    return kwargs",
            "args": (),
            "expect_alias": True,
        },
        {
            "name": "alias_multiline",
            "src": "def alias_multiline(\n    a,\n    kwargs=\"(a, b)\",  # comment\n    **kwargs,\n):\n    return kwargs",
            "args": (1,),
            "expect_alias": True,
        },
        {
            "name": "alias_docstring",
            "src": "def alias_docstring(kwargs=1, **kwargs):\n    \"\"\"docstring\"\"\"\n    return kwargs",
            "args": (),
            "expect_alias": True,
            "expect_docstring_alias": True,
        },
        {
            "name": "alias_single_line",
            "src": "def alias_single_line(kwargs=1, **kwargs): return kwargs",
            "args": (),
            "expect_alias": True,
        },
        {
            "name": "alias_inline_hang",
            "src": "def alias_inline_hang(\n        a,\n        kwargs=1,\n        **kwargs,\n        ): return kwargs",
            "args": (1,),
            "expect_alias": True,
        },
        {
            "name": "alias_async",
            "src": "async def alias_async(kwargs=1, **kwargs):\n    return kwargs",
            "args": (),
            "expect_alias": True,
            "is_async": True,
        },
        {
            "name": "alias_async_single_line",
            "src": "async def alias_async_single_line(kwargs=1, **kwargs): return kwargs",
            "args": (),
            "expect_alias": True,
            "is_async": True,
        },
        {
            "name": "alias_type_params",
            "src": "def alias_type_params[T](kwargs=1, **kwargs):\n    return kwargs",
            "args": (),
            "expect_alias": True,
        },
        {
            "name": "no_alias_kwargs_only",
            "src": "def no_alias_kwargs_only(kwargs=1):\n    return kwargs",
            "args": (),
            "expect_alias": False,
        },
        {
            "name": "no_alias_kwargs_kwarg",
            "src": "def no_alias_kwargs_kwarg(**kwargs):\n    return kwargs",
            "args": (),
            "expect_alias": False,
        },
        {
            "name": "no_alias_varargs",
            "src": "def no_alias_varargs(*args, **kwargs):\n    return args",
            "args": (1, 2),
            "expect_alias": False,
        },
        {
            "name": "no_alias_annotated",
            "src": "def no_alias_annotated(kwargs: dict = None):\n    return kwargs",
            "args": (None,),
            "expect_alias": False,
        },
        {
            "name": "no_alias_type_params",
            "src": "def no_alias_type_params[T](a, **kwargs):\n    return a",
            "args": (1,),
            "expect_alias": False,
        },
    ]

    results = []
    for case in cases:
        try:
            results.append(run_case(case))
        except Exception as e:
            results.append({"name": case["name"], "status": "error", "error": repr(e)})

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"kwargs_signature_harness_{run_id}.json"
    log_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"Wrote {log_path}")
    print(results)


if __name__ == "__main__":
    main()

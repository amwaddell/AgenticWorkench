#!/usr/bin/env python3
"""
map_architecture.py — Generate an LLM-friendly architectural overview of a Python project.

Usage:
    python map_architecture.py [ROOT_DIR] [--out OUTPUT_FILE] [--src SRC_DIR] [--skip PATTERN...]

Examples:
    python map_architecture.py .                                  # scan current dir
    python map_architecture.py . --out runs/reports/inventory.md  # custom output path
    python map_architecture.py . --src src/workbench              # only scan src/workbench
    python map_architecture.py . --skip .venv __pycache__ node_modules

Output:  A single Markdown file designed to be pasted into an LLM context window.
         Contains: file inventory, imports, classes, functions, docstrings, cross-module
         dependency graph, entrypoint analysis, and dead-code candidates.

Requirements: Python 3.9+ (stdlib only — no third-party packages needed).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────


@dataclass
class FuncInfo:
    name: str
    lineno: int
    args: list[str]
    decorators: list[str]
    docstring: str | None
    is_async: bool = False

    @property
    def visibility(self) -> str:
        if self.name.startswith("__") and self.name.endswith("__"):
            return "dunder"
        if self.name.startswith("_"):
            return "private"
        return "public"


@dataclass
class ClassInfo:
    name: str
    lineno: int
    bases: list[str]
    decorators: list[str]
    docstring: str | None
    methods: list[FuncInfo] = field(default_factory=list)


@dataclass
class FileInfo:
    path: str  # relative to project root
    module: str  # dotted module name
    docstring: str | None
    imports: list[str]  # raw import strings
    import_modules: list[str]  # resolved module names (for dep graph)
    classes: list[ClassInfo]
    functions: list[FuncInfo]  # module-level functions only
    global_vars: list[str]  # module-level assignments (ALL_CAPS or __dunder__)
    lines_total: int
    lines_code: int  # non-blank, non-comment


# ──────────────────────────────────────────────────────────────────────
# AST extraction
# ──────────────────────────────────────────────────────────────────────


def _get_decorators(node: ast.AST) -> list[str]:
    decorators = []
    for dec in getattr(node, "decorator_list", []):
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            decorators.append(ast.dump(dec))  # fallback
            # try to reconstruct dotted name
            parts = []
            current = dec
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                decorators[-1] = ".".join(reversed(parts))
        elif isinstance(dec, ast.Call):
            # decorator with arguments — get the function name
            func = dec.func
            if isinstance(func, ast.Name):
                decorators.append(f"{func.id}(...)")
            elif isinstance(func, ast.Attribute):
                parts = []
                current = func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                decorators.append(f"{'.'.join(reversed(parts))}(...)")
            else:
                decorators.append("@?")
    return decorators


def _get_func_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = []
    for arg in node.args.args:
        annotation = ""
        if arg.annotation:
            try:
                annotation = f": {ast.unparse(arg.annotation)}"
            except Exception:
                annotation = ": ?"
        args.append(f"{arg.arg}{annotation}")
    return args


def _extract_imports(tree: ast.Module) -> tuple[list[str], list[str]]:
    """Return (raw_import_strings, resolved_module_names)."""
    raw = []
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                raw.append(
                    f"import {alias.name}"
                    + (f" as {alias.asname}" if alias.asname else "")
                )
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = ", ".join(
                a.name + (f" as {a.asname}" if a.asname else "") for a in node.names
            )
            level_dots = "." * (node.level or 0)
            raw.append(f"from {level_dots}{mod} import {names}")
            if node.level and node.level > 0:
                modules.append(f"(relative) {level_dots}{mod}")
            else:
                modules.append(mod)
    return raw, modules


def _extract_global_vars(tree: ast.Module) -> list[str]:
    """Extract notable module-level variable assignments."""
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if name.isupper() or (
                        name.startswith("__") and name.endswith("__")
                    ):
                        names.append(name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.isupper() or (name.startswith("__") and name.endswith("__")):
                names.append(name)
    return names


def _count_code_lines(source: str) -> int:
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def analyze_file(filepath: Path, root: Path) -> FileInfo | None:
    """Parse a single Python file and extract its structure."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return None

    rel = filepath.relative_to(root)
    module = str(rel.with_suffix("")).replace(os.sep, ".")

    raw_imports, import_modules = _extract_imports(tree)
    global_vars = _extract_global_vars(tree)

    classes: list[ClassInfo] = []
    functions: list[FuncInfo] = []

    for node in tree.body:
        if isinstance(node, (ast.ClassDef,)):
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(
                        FuncInfo(
                            name=item.name,
                            lineno=item.lineno,
                            args=_get_func_args(item),
                            decorators=_get_decorators(item),
                            docstring=ast.get_docstring(item),
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                        )
                    )
            bases = []
            for base in node.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    bases.append("?")
            classes.append(
                ClassInfo(
                    name=node.name,
                    lineno=node.lineno,
                    bases=bases,
                    decorators=_get_decorators(node),
                    docstring=ast.get_docstring(node),
                    methods=methods,
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                FuncInfo(
                    name=node.name,
                    lineno=node.lineno,
                    args=_get_func_args(node),
                    decorators=_get_decorators(node),
                    docstring=ast.get_docstring(node),
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                )
            )

    return FileInfo(
        path=str(rel),
        module=module,
        docstring=ast.get_docstring(tree),
        imports=raw_imports,
        import_modules=import_modules,
        classes=classes,
        functions=functions,
        global_vars=global_vars,
        lines_total=len(source.splitlines()),
        lines_code=_count_code_lines(source),
    )


# ──────────────────────────────────────────────────────────────────────
# Project scanning
# ──────────────────────────────────────────────────────────────────────

DEFAULT_SKIP = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".egg-info",
    "egg-info",
    "dist",
    "build",
    ".eggs",
    "wikiextractor",
}


def find_python_files(
    root: Path, src_filter: str | None, skip_patterns: set[str]
) -> list[Path]:
    """Walk the project tree and return all .py files, respecting skip patterns."""
    files = []
    scan_root = root / src_filter if src_filter else root
    for dirpath, dirnames, filenames in os.walk(scan_root):
        # prune skipped directories in-place
        dirnames[:] = [
            d
            for d in dirnames
            if d not in skip_patterns and not d.endswith(".egg-info")
        ]
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                files.append(Path(dirpath) / fname)
    return files


# ──────────────────────────────────────────────────────────────────────
# Dependency graph (internal modules only)
# ──────────────────────────────────────────────────────────────────────


def build_dependency_graph(files: list[FileInfo]) -> dict[str, set[str]]:
    """Build a graph of internal module dependencies."""
    all_modules = {f.module for f in files}
    # Also track package prefixes so we can match "from workbench.retrieval import X"
    all_prefixes = set()
    for m in all_modules:
        parts = m.split(".")
        for i in range(1, len(parts) + 1):
            all_prefixes.add(".".join(parts[:i]))

    graph: dict[str, set[str]] = defaultdict(set)
    for f in files:
        for imp in f.import_modules:
            if imp.startswith("(relative)"):
                # resolve relative import
                imp_clean = imp.replace("(relative) ", "")
                # try matching against known modules
                for mod in all_modules:
                    if mod.endswith(imp_clean.lstrip(".")):
                        graph[f.module].add(mod)
                        break
            else:
                # check if it's an internal module
                if imp in all_prefixes:
                    # find the most specific matching module
                    best = imp
                    if imp not in all_modules:
                        # might be importing from a package — find parent
                        candidates = [m for m in all_modules if m.startswith(imp)]
                        if candidates:
                            best = min(candidates, key=len)
                    graph[f.module].add(best)
    return dict(graph)


def find_unreferenced_modules(
    files: list[FileInfo], graph: dict[str, set[str]]
) -> list[str]:
    """Find modules that are never imported by any other module."""
    all_modules = {f.module for f in files}
    referenced: set[str] = set()
    for deps in graph.values():
        referenced.update(deps)

    # entrypoints (scripts, tests, __init__) are expected to be unreferenced
    unreferenced = []
    for m in sorted(all_modules - referenced):
        # skip __init__ and obvious entrypoints
        basename = m.split(".")[-1]
        if basename in ("__init__",):
            continue
        unreferenced.append(m)
    return unreferenced


# ──────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────


def _trunc_docstring(doc: str | None, max_chars: int = 200) -> str:
    if not doc:
        return "(no docstring)"
    doc = " ".join(doc.split())  # collapse whitespace
    if len(doc) > max_chars:
        return doc[:max_chars] + "…"
    return doc


def _format_func(fn: FuncInfo, indent: str = "  ") -> str:
    prefix = "async " if fn.is_async else ""
    decs = "".join(f"{indent}@{d}\n" for d in fn.decorators)
    args_str = ", ".join(fn.args)
    vis = f"[{fn.visibility}]" if fn.visibility != "public" else ""
    doc = f'\n{indent}  """{_trunc_docstring(fn.docstring)}"""' if fn.docstring else ""
    return f"{decs}{indent}{prefix}def {fn.name}({args_str}) {vis}{doc}"


def generate_report(
    files: list[FileInfo],
    graph: dict[str, set[str]],
    root: Path,
    scan_info: str,
) -> str:
    """Generate the full Markdown report."""
    sections = []

    # ── Header ──
    sections.append(
        textwrap.dedent(f"""\
    # Project Architecture Overview
    
    > Auto-generated by `map_architecture.py`
    > Root: `{root.resolve()}`
    > Scanned: {scan_info}
    > Files analyzed: {len(files)}
    > Total lines of code: {sum(f.lines_code for f in files):,}
    
    ---
    """)
    )

    # ── Table of contents / summary ──
    sections.append("## Quick Summary\n")
    # group by top-level package
    packages: dict[str, list[FileInfo]] = defaultdict(list)
    for f in files:
        parts = f.path.split(os.sep)
        pkg = parts[0] if len(parts) > 1 else "(root)"
        packages[pkg].append(f)

    for pkg in sorted(packages):
        pkg_files = packages[pkg]
        total_loc = sum(f.lines_code for f in pkg_files)
        total_classes = sum(len(f.classes) for f in pkg_files)
        total_funcs = sum(len(f.functions) for f in pkg_files)
        sections.append(
            f"- **{pkg}/**: {len(pkg_files)} files, {total_loc} LoC, "
            f"{total_classes} classes, {total_funcs} module-level functions"
        )
    sections.append("")

    # ── Per-file inventory ──
    sections.append("---\n## File-by-File Inventory\n")
    for f in sorted(files, key=lambda x: x.path):
        sections.append(f"### `{f.path}`")
        sections.append(f"- **Module**: `{f.module}`")
        sections.append(f"- **Lines**: {f.lines_total} total, {f.lines_code} code")
        if f.docstring:
            sections.append(f"- **Purpose**: {_trunc_docstring(f.docstring, 300)}")

        if f.global_vars:
            sections.append(f"- **Constants/Globals**: {', '.join(f.global_vars)}")

        if f.imports:
            sections.append("- **Imports**:")
            for imp in f.imports:
                sections.append(f"  - `{imp}`")

        for cls in f.classes:
            bases = f"({', '.join(cls.bases)})" if cls.bases else ""
            decs = ", ".join(f"@{d}" for d in cls.decorators)
            if decs:
                decs = f" [{decs}]"
            sections.append(f"- **class `{cls.name}`**{bases}{decs}")
            if cls.docstring:
                sections.append(f"  - {_trunc_docstring(cls.docstring, 300)}")
            for m in cls.methods:
                vis = f" [{m.visibility}]" if m.visibility != "public" else ""
                async_tag = " (async)" if m.is_async else ""
                args_str = ", ".join(m.args)
                sections.append(f"  - `{m.name}({args_str})`{vis}{async_tag}")
                if m.docstring:
                    sections.append(f"    - {_trunc_docstring(m.docstring, 150)}")

        for fn in f.functions:
            vis = f" [{fn.visibility}]" if fn.visibility != "public" else ""
            async_tag = " (async)" if fn.is_async else ""
            args_str = ", ".join(fn.args)
            decs = ", ".join(f"@{d}" for d in fn.decorators)
            if decs:
                decs = f" [{decs}]"
            sections.append(f"- **`def {fn.name}({args_str})`**{vis}{async_tag}{decs}")
            if fn.docstring:
                sections.append(f"  - {_trunc_docstring(fn.docstring, 200)}")

        sections.append("")

    # ── Internal dependency graph ──
    sections.append("---\n## Internal Dependency Graph\n")
    sections.append("Each line shows: `module → [depends on]`\n")
    all_modules_sorted = sorted({f.module for f in files})
    for mod in all_modules_sorted:
        deps = graph.get(mod, set())
        if deps:
            dep_list = ", ".join(sorted(deps))
            sections.append(f"- `{mod}` → {dep_list}")
    sections.append("")

    # ── Reverse dependencies (who imports this module?) ──
    sections.append("### Reverse Dependencies (who imports this?)\n")
    reverse: dict[str, set[str]] = defaultdict(set)
    for mod, deps in graph.items():
        for dep in deps:
            reverse[dep].add(mod)
    for mod in all_modules_sorted:
        importers = reverse.get(mod, set())
        if importers:
            sections.append(f"- `{mod}` ← imported by: {', '.join(sorted(importers))}")
    sections.append("")

    # ── Unreferenced modules ──
    unreferenced = find_unreferenced_modules(files, graph)
    if unreferenced:
        sections.append("### Potentially Unused Modules\n")
        sections.append("These modules are never imported by any other scanned module.")
        sections.append("Some may be entrypoints (scripts, tests) which is expected.\n")
        for m in unreferenced:
            sections.append(f"- `{m}`")
        sections.append("")

    # ── External dependencies summary ──
    sections.append("---\n## External Dependencies\n")
    sections.append("Third-party packages imported across the project:\n")
    internal_prefixes = set()
    for f in files:
        parts = f.module.split(".")
        if parts:
            internal_prefixes.add(parts[0])
    # also add common stdlib-ish prefixes we shouldn't list
    stdlib_common = {
        "os",
        "sys",
        "re",
        "json",
        "csv",
        "math",
        "time",
        "datetime",
        "pathlib",
        "collections",
        "functools",
        "itertools",
        "typing",
        "dataclasses",
        "abc",
        "logging",
        "unittest",
        "pytest",
        "argparse",
        "textwrap",
        "hashlib",
        "uuid",
        "copy",
        "io",
        "contextlib",
        "enum",
        "operator",
        "string",
        "tempfile",
        "shutil",
        "subprocess",
        "threading",
        "multiprocessing",
        "ast",
        "inspect",
        "importlib",
        "pkgutil",
        "warnings",
        "traceback",
        "pprint",
        "glob",
        "fnmatch",
        "struct",
        "pickle",
        "sqlite3",
        "html",
        "xml",
        "http",
        "urllib",
        "email",
        "socket",
        "ssl",
        "signal",
        "concurrent",
        "asyncio",
        "builtins",
        "types",
        "numbers",
        "decimal",
        "fractions",
        "random",
        "statistics",
        "bisect",
        "heapq",
        "array",
        "weakref",
        "codecs",
        "unicodedata",
        "locale",
        "calendar",
        "configparser",
        "tomllib",
        "tomli",
        "secrets",
        "hmac",
        "__future__",
        "typing_extensions",
        "zlib",
        "gzip",
        "bz2",
        "lzma",
        "zipfile",
        "tarfile",
        "platform",
        "sysconfig",
        "ctypes",
    }

    external: dict[str, set[str]] = defaultdict(
        set
    )  # package → set of importing modules
    for f in files:
        for imp in f.import_modules:
            if imp.startswith("(relative)"):
                continue
            top = imp.split(".")[0]
            if top not in internal_prefixes and top not in stdlib_common and top:
                external[top].add(f.module)

    for pkg in sorted(external):
        importers = sorted(external[pkg])
        sections.append(f"- **{pkg}** — used by: {', '.join(importers)}")
    sections.append("")

    # ── Entrypoints ──
    sections.append("---\n## Entrypoints\n")
    sections.append("Files likely used as entrypoints (scripts, __main__, etc.):\n")
    for f in sorted(files, key=lambda x: x.path):
        is_script = "scripts" in f.path.split(os.sep)
        is_example = "examples" in f.path.split(os.sep)
        is_test = f.path.split(os.sep)[-1].startswith("test_")
        has_main_guard = False
        try:
            source = (root / f.path).read_text(encoding="utf-8", errors="replace")
            has_main_guard = "if __name__" in source
        except Exception:
            pass

        if is_script or is_example or has_main_guard:
            tag = ""
            if is_script:
                tag = " [script]"
            elif is_example:
                tag = " [example]"
            elif is_test:
                tag = " [test]"
            sections.append(f"- `{f.path}`{tag}")
            # show what it imports from within the project
            internal_deps = graph.get(f.module, set())
            if internal_deps:
                sections.append(f"  - Uses: {', '.join(sorted(internal_deps))}")
    sections.append("")

    # ── ASCII dependency diagram (simplified) ──
    sections.append("---\n## Simplified Package Dependency Diagram\n")
    sections.append("```")
    # aggregate to package level
    pkg_deps: dict[str, set[str]] = defaultdict(set)
    for mod, deps in graph.items():
        parts = mod.split(".")
        # use 2-level package name if available
        src_pkg = ".".join(parts[:3]) if len(parts) >= 3 else mod
        for dep in deps:
            dep_parts = dep.split(".")
            dep_pkg = ".".join(dep_parts[:3]) if len(dep_parts) >= 3 else dep
            if src_pkg != dep_pkg:
                pkg_deps[src_pkg].add(dep_pkg)

    for pkg in sorted(pkg_deps):
        for dep in sorted(pkg_deps[pkg]):
            sections.append(f"  {pkg}  -->  {dep}")
    sections.append("```\n")

    # ── Footer ──
    sections.append("---\n*End of architecture overview.*\n")

    return "\n".join(sections)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate an LLM-friendly architectural overview of a Python project."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output file path (default: <root>/runs/reports/architecture_overview.md)",
    )
    parser.add_argument(
        "--src",
        "-s",
        default=None,
        help="Subdirectory to scan (e.g. 'src/workbench'). If not set, scans all .py files.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Additional directory names to skip.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        default=True,
        help="Include test files in the scan (default: True).",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    skip = DEFAULT_SKIP | set(args.skip)

    # Determine scan scope
    scan_dirs = []
    if args.src:
        scan_dirs.append(args.src)
        scan_info = f"`{args.src}/`"
    else:
        # scan common Python directories
        candidates = ["src", "scripts", "tests", "examples", "notebooks"]
        for c in candidates:
            if (root / c).is_dir():
                scan_dirs.append(c)
        # also check for .py files in root
        if any(root.glob("*.py")):
            scan_dirs.append(None)  # sentinel for root-level files
        scan_info = ", ".join(f"`{d or '.'}/`" for d in scan_dirs)

    # Collect files
    all_py_files: list[Path] = []
    for d in scan_dirs:
        if d is None:
            all_py_files.extend(sorted(root.glob("*.py")))
        else:
            all_py_files.extend(find_python_files(root, d, skip))

    # Deduplicate
    seen = set()
    unique_files = []
    for f in all_py_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    print(f"Scanning {len(unique_files)} Python files in {root}...")

    # Analyze
    file_infos: list[FileInfo] = []
    for fp in unique_files:
        info = analyze_file(fp, root)
        if info:
            file_infos.append(info)

    print(f"Successfully parsed {len(file_infos)} / {len(unique_files)} files.")

    # Build dependency graph
    graph = build_dependency_graph(file_infos)

    # Generate report
    report = generate_report(file_infos, graph, root, scan_info)

    # Write output
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = root / "runs" / "reports" / "architecture_overview.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"\nArchitecture overview written to: {out_path}")
    print(f"  Size: {len(report):,} characters ({len(report.splitlines()):,} lines)")
    print("\nTip: paste this into an LLM context window alongside specific code")
    print("     files you want to modify for targeted assistance.")


if __name__ == "__main__":
    main()

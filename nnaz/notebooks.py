"""Generate a Jupyter notebook for every lesson from its ``lesson.py``.

The notebook is built directly from the lesson source, so the two can never
disagree:

* the lesson's section of the README (theory and maths) becomes the first markdown cell,
* every top-level function becomes its own code cell, preceded by a markdown
  cell with the section title taken from the function's ``banner(...)`` call,
* the body of ``main()`` becomes one cell per step, each followed by
  ``show_new_figures()`` so the figures saved by that step are displayed inline.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from .common import DOCS, ROOT

NB_DIR = ROOT / "notebooks"


def show_new_figures(since: float, lesson: int):
    """Display (in Jupyter) every figure/GIF of ``docs/lessonNN`` newer than ``since``."""
    from IPython.display import Image, display

    d = DOCS / f"lesson{lesson:02d}"
    for p in sorted(d.glob("*"), key=lambda p: p.stat().st_mtime):
        if p.stat().st_mtime >= since and p.suffix in (".png", ".gif"):
            display(Image(filename=str(p)))


def _readme_section(lesson: int) -> str:
    text = (ROOT / "README.md").read_text()
    m = re.search(rf"^## Lesson {lesson:02d}.*?(?=^---$)", text, re.S | re.M)
    if not m:
        return ""
    sec = m.group(0)
    sec = re.sub(r"<!-- results:\d+ -->.*?<!-- /results:\d+ -->", "*(see the results table in the README)*", sec, flags=re.S)
    return sec.replace('src="docs/', 'src="../docs/').replace("](docs/", "](../docs/") \
              .replace("](lessons/", "](../lessons/").replace("](nnaz/", "](../nnaz/") \
              .replace("](notebooks/", "](")


def build_notebook(lesson_dir: Path) -> Path:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    src = (lesson_dir / "lesson.py").read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    lesson = int(lesson_dir.name[:2])

    def seg(node):
        start = node.lineno - 1
        if getattr(node, "decorator_list", None):
            start = node.decorator_list[0].lineno - 1
        return "\n".join(lines[start:node.end_lineno])

    cells = []
    title = f"Lesson {lesson:02d}"
    doc = ast.get_docstring(tree) or ""
    head, _, rest = doc.partition("\n")
    cells.append(new_markdown_cell(f"# {head}\n\n{rest.replace('Run:', 'Script version:')}"))
    theory = _readme_section(lesson)
    if theory:
        cells.append(new_markdown_cell(theory))

    setup = ["%matplotlib inline",
             "import sys, time",
             "from pathlib import Path",
             "sys.path.insert(0, str(Path.cwd().parent))  # make `nnaz` importable",
             "from nnaz.notebooks import show_new_figures"]
    body = []
    for node in tree.body[1:]:
        code = seg(node)
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            continue
        if isinstance(node, ast.If):  # the `if __name__ == "__main__"` guard
            continue
        if isinstance(node, ast.Expr) and "sys.path.insert" in code:
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if body:
                cells.append(new_code_cell("\n".join(setup + [""] + body) if setup else "\n".join(body)))
                setup, body = None, []
            m = re.search(r'banner\("([^"]+)"\)', code)
            fdoc = ast.get_docstring(node)
            if m:
                cells.append(new_markdown_cell(f"## {m.group(1)}"))
            elif fdoc and isinstance(node, ast.ClassDef):
                cells.append(new_markdown_cell(f"**Helper `{node.name}`**: {fdoc}"))
            cells.append(new_code_cell(code))
        else:
            body.append(code)
    if body:
        cells.append(new_code_cell("\n".join((setup or []) + [""] + body)))

    # main(): one cell per statement, showing the figures it produced
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    cells.append(new_markdown_cell(f"## Run the lesson\n\nSet `quick = True` for a fast pass "
                                   f"(fewer epochs; numbers will differ from the README)."))
    cells.append(new_code_cell("quick = False"))
    for stmt in main.body:
        code = seg(stmt)
        code = "\n".join(l[4:] if l.startswith("    ") else l for l in code.splitlines())
        if "save_results" in code:
            code = "_ = " + code
        if isinstance(stmt, ast.Return):
            cells.append(new_code_cell("res  # all numbers of this run"))
            continue
        if "_part(" in code:
            code = f"t0 = time.time()\n{code}\nshow_new_figures(t0, LESSON)"
        cells.append(new_code_cell(code))

    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}})
    NB_DIR.mkdir(exist_ok=True)
    out = NB_DIR / f"{lesson_dir.name}.ipynb"
    nbformat.write(nb, out)
    return out


def execute_notebook(path: Path, timeout: int = 1800) -> None:
    """Run a notebook in place (outputs are stored, so GitHub renders them)."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)

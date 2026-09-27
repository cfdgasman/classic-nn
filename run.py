"""Regenerate every figure, GIF, results table and notebook of the course.

    python run.py                    # all lessons (CPU, a few minutes each)
    python run.py --lessons 1 2 3    # a subset
    python run.py --quick            # fast smoke pass (fewer epochs; numbers differ)
    python run.py --notebooks-only   # just rebuild the notebooks from lesson.py
    python run.py --execute          # also execute the notebooks (stores their outputs)

Outputs: docs/lessonNN/*.png|gif, docs/results/lessonNN.json, the tables inside
README.md (between <!-- results:NN --> markers) and notebooks/NN_topic.ipynb.
"""
from __future__ import annotations

import argparse
import runpy
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LESSONS = sorted(p for p in (ROOT / "lessons").iterdir() if (p / "lesson.py").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lessons", type=int, nargs="*", help="lesson numbers (default: all)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--notebooks-only", action="store_true")
    ap.add_argument("--execute", action="store_true", help="execute the generated notebooks")
    args = ap.parse_args()

    chosen = [p for p in LESSONS if not args.lessons or int(p.name[:2]) in args.lessons]
    timings = {}
    if not args.notebooks_only:
        for p in chosen:
            print(f"\n######## {p.name} ########")
            t0 = time.time()
            mod = runpy.run_path(str(p / "lesson.py"), run_name="lesson")
            mod["main"](quick=args.quick)
            timings[p.name] = time.time() - t0
        if not args.quick:  # never write smoke-run numbers into the README
            from nnaz.report import update_readme

            update_readme()

    from nnaz.notebooks import build_notebook, execute_notebook

    for p in chosen:
        nb = build_notebook(p)
        if args.execute:
            print(f"executing {nb.name} ...")
            execute_notebook(nb)
        print(f"  notebook {nb.relative_to(ROOT)}")

    if timings:
        print("\nwall-clock time per lesson:")
        for k, v in timings.items():
            print(f"  {k:28s} {v:6.1f} s")


if __name__ == "__main__":
    main()

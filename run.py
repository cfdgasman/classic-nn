"""Regenerate every figure, GIF, results table and notebook of the course.

    python run.py                    # run all lessons as scripts, update README tables, build notebooks
    python run.py --lessons 1 2 3    # a subset
    python run.py --quick            # fast smoke pass (fewer epochs; README tables NOT updated)
    python run.py --execute          # run each lesson THROUGH its notebook (stores the outputs in
                                     #   notebooks/*.ipynb) - one pass produces notebooks, figures and tables
    python run.py --notebooks-only   # just rebuild the (unexecuted) notebooks from lesson.py

Outputs: docs/lessonNN/*.png|gif, docs/results/lessonNN.json, the tables inside
README.md (between <!-- results:NN --> markers) and notebooks/NN_topic.ipynb.
Lessons run one after another on purpose: each uses all CPU cores.
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
    ap.add_argument("--execute", action="store_true", help="run the lessons by executing their notebooks")
    args = ap.parse_args()

    from nnaz.notebooks import build_notebook, execute_notebook

    chosen = [p for p in LESSONS if not args.lessons or int(p.name[:2]) in args.lessons]
    timings = {}
    for p in chosen:
        print(f"\n######## {p.name} ########", flush=True)
        t0 = time.time()
        if args.execute:
            nb = build_notebook(p)
            execute_notebook(nb)
        elif not args.notebooks_only:
            mod = runpy.run_path(str(p / "lesson.py"), run_name="lesson")
            mod["main"](quick=args.quick)
            build_notebook(p)
        else:
            build_notebook(p)
        timings[p.name] = time.time() - t0

    if not args.quick and not args.notebooks_only:  # never write smoke-run numbers into the README
        from nnaz.report import update_readme

        update_readme()

    print("\nwall-clock time per lesson:")
    for k, v in timings.items():
        print(f"  {k:28s} {v:6.1f} s")


if __name__ == "__main__":
    main()

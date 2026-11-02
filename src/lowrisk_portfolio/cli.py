"""Run the final-report pipeline without modifying the supplied reference results."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .paths import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    parser.add_argument(
        "--figures-only", action="store_true", help="Redraw figures from supplied reference tables."
    )
    parser.add_argument(
        "--certify",
        action="store_true",
        help="Also regenerate expensive independent solver certificates.",
    )
    parser.add_argument(
        "--pdf", action="store_true", help="Compile the regenerated report with pdflatex."
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    reference = (ROOT / "results/reference").resolve()
    if output == reference or reference in output.parents or output in reference.parents:
        parser.error("Choose an output directory separate from results/reference and its parents.")
    if not args.figures_only and shutil.which("bibtex") is None:
        parser.error(
            "The full report build requires BibTeX and IEEEtran.bst from a TeX distribution."
        )
    if args.pdf and shutil.which("pdflatex") is None:
        parser.error("--pdf requires pdflatex.")
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, LOWRISK_OUTPUT_DIR=str(output))
    environment.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    started = time.perf_counter()
    if args.figures_only:
        for name in (
            "monthly_returns.csv",
            "benchmark_returns.csv",
            "regime_coverage.csv",
            "regime_metrics.csv",
        ):
            shutil.copy2(reference / name, output / name)
        stages = ["figures"]
    else:
        stages = [
            "experiment",
            "mechanisms",
            "selection_comparison",
            "report",
            "market_inference",
            "verify_benchmarks",
        ]
        if args.certify:
            stages.extend(["verify_solvers", "verify_numerics"])
    manifest = {
        "status": "running",
        "mode": "figures" if args.figures_only else "full",
        "stages": [],
    }
    manifest_path = output / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for stage in stages:
            print(f"\n[{stage}]", flush=True)
            stage_start = time.perf_counter()
            source_hash = hashlib.sha256(
                Path(__file__).with_name(stage + ".py").read_bytes()
            ).hexdigest()
            subprocess.run(
                [sys.executable, "-m", f"lowrisk_portfolio.{stage}"], env=environment, check=True
            )
            manifest["stages"].append(
                {
                    "name": stage,
                    "seconds": round(time.perf_counter() - stage_start, 3),
                    "source_sha256": source_hash,
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if args.pdf:
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                    cwd=output / "paper",
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        manifest["seconds"] = round(time.perf_counter() - started, 3)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed in {manifest['seconds']:.1f} seconds. Results: {output}")


if __name__ == "__main__":
    main()

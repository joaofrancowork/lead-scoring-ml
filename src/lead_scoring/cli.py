"""lead-scoring download | train | serve"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lead_scoring.data import DEFAULT_PATH, download, load, random_split, time_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lead-scoring")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="Fetch the UCI Bank Marketing dataset")
    dl.add_argument("--path", type=Path, default=DEFAULT_PATH)
    tr = sub.add_parser("train", help="Train, select, calibrate and save the model")
    tr.add_argument("--data", type=Path, default=DEFAULT_PATH)
    tr.add_argument("--out", type=Path, default=Path("artifacts/model"))
    tr.add_argument("--split", choices=["time", "random"], default="time")
    tr.add_argument("--features", choices=["all", "client"], default="client")
    tr.add_argument("--fast", action="store_true", help="Fewer boosting rounds (CI)")
    dr = sub.add_parser("drift", help="PSI drift report: training period vs the latest contacts")
    dr.add_argument("--data", type=Path, default=DEFAULT_PATH)
    dr.add_argument("--features", choices=["all", "client"], default="all")
    bt = sub.add_parser("backtest", help="Rolling-origin backtest across consecutive periods")
    bt.add_argument("--data", type=Path, default=DEFAULT_PATH)
    bt.add_argument("--features", choices=["all", "client"], default="client")
    bt.add_argument("--folds", type=int, default=5)
    bt.add_argument("--fast", action="store_true")
    sv = sub.add_parser("serve", help="Run the scoring API")
    sv.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "download":
        print(download(args.path))
        return 0
    if args.command == "serve":  # pragma: no cover
        import uvicorn

        from lead_scoring.serving import create_app

        uvicorn.run(create_app(), host="0.0.0.0", port=args.port)
        return 0

    if args.command == "drift":
        from lead_scoring.features import input_columns
        from lead_scoring.monitoring import drift_report

        train_df, _, test_df = time_split(load(download(args.data)))
        report = drift_report(train_df, test_df, input_columns(args.features))
        print(report.to_markdown())
        print(f"\nalerts: {', '.join(report.alerts) or 'none'}")
        return 0

    if args.command == "backtest":
        from lead_scoring.backtest import backtest, to_markdown

        results = backtest(
            load(download(args.data)), args.features, n_folds=args.folds, fast=args.fast
        )
        print(to_markdown(results))
        return 0

    from lead_scoring.train import train

    df = load(download(args.data))
    splits = time_split(df) if args.split == "time" else random_split(df)
    result = train(
        *splits,
        out_dir=args.out,
        split_name=args.split,
        fast=args.fast,
        feature_set=args.features,
    )
    print(json.dumps({"best": result.best, "test": result.test}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

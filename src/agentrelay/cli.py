"""Command-line entry points for validation, provenance, and aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import StorageLayout, load_json_config, validate_experiment_config
from .datasets import dataset_provenance, get_dataset_spec, load_public_dataset
from .metrics import aggregate_trajectories, read_jsonl
from .schema import RelayStatePacket
from .validation import PacketValidator


def _validate_packet(args: argparse.Namespace) -> int:
    packet = RelayStatePacket.from_json(Path(args.packet).read_text(encoding="utf-8"))
    report = PacketValidator().validate(packet)
    print(json.dumps({"valid": report.valid, "issues": [issue.__dict__ for issue in report.issues]}))
    return 0 if report.valid else 2


def _aggregate(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    result = aggregate_trajectories(rows)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _download_dataset(args: argparse.Namespace) -> int:
    spec = get_dataset_spec(args.name, revision=args.revision)
    storage = StorageLayout(Path(args.data_root).resolve())
    dataset = load_public_dataset(spec, split=args.split, storage=storage)
    print(
        json.dumps(
            {
                "rows": len(dataset),
                "provenance": dataset_provenance(spec, args.split),
                "cache": str(storage.datasets),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _check_config(args: argparse.Namespace) -> int:
    config = load_json_config(args.config)
    validate_experiment_config(config, allow_unlocked=args.allow_unlocked)
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _local_smoke(args: argparse.Namespace) -> int:
    from .local_smoke import run_local_smoke

    result = run_local_smoke(
        args.config,
        limit=args.limit,
        subset=args.subset,
        command=tuple(sys.argv),
        required_gpu_name=args.required_gpu_name,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    summary["run_directory"] = str(result.run_directory)
    summary["manifest_path"] = str(result.manifest_path)
    summary["outputs_path"] = str(result.outputs_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _verify_run(args: argparse.Namespace) -> int:
    from .run_audit import audit_run

    report = audit_run(
        args.run_directory,
        required_gpu_name=args.required_gpu_name,
        require_diagnostic_only=args.require_diagnostic_only,
        project_root=args.project_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentrelay")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-packet")
    validate.add_argument("packet")
    validate.set_defaults(func=_validate_packet)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("input")
    aggregate.add_argument("--output")
    aggregate.set_defaults(func=_aggregate)

    download = subparsers.add_parser("download-dataset")
    download.add_argument("--name", required=True)
    download.add_argument("--revision", required=True, help="immutable HF commit hash")
    download.add_argument("--split", required=True)
    download.add_argument("--data-root", default="/root/autodl-tmp/AgentRelay")
    download.set_defaults(func=_download_dataset)

    check_config = subparsers.add_parser("check-config")
    check_config.add_argument("config")
    check_config.add_argument(
        "--allow-unlocked",
        action="store_true",
        help="validate a revision template without authorizing it for execution",
    )
    check_config.set_defaults(func=_check_config)

    local_smoke = subparsers.add_parser(
        "local-smoke",
        help="run a non-evidence public-data native-inference diagnostic",
    )
    local_smoke.add_argument("config", help="locked local-smoke JSON config")
    local_smoke.add_argument("--limit", type=int, default=1)
    local_smoke.add_argument("--subset")
    local_smoke.add_argument("--required-gpu-name", default="4080")
    local_smoke.set_defaults(func=_local_smoke)

    verify_run = subparsers.add_parser(
        "verify-run",
        help="independently verify run hashes, packet provenance, and environment metadata",
    )
    verify_run.add_argument("run_directory")
    verify_run.add_argument("--required-gpu-name", default="")
    verify_run.add_argument("--require-diagnostic-only", action="store_true")
    verify_run.add_argument(
        "--project-root",
        help="also require the current runtime source tree to match the manifest",
    )
    verify_run.set_defaults(func=_verify_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

from pathlib import Path

from src.utils.run_manager import check_required_outputs


REQUIRED_OUTPUTS_FULL = [
    "raw_logs/node_log.csv",
    "raw_logs/link_log.csv",
    "raw_logs/service_log.csv",
    "raw_logs/path_log.csv",
    "raw_logs/sla_log.csv",
    "dataset/train.pkl",
    "dataset/val.pkl",
    "dataset/test.pkl",
    "checkpoints/lstm_best.pth",
    "checkpoints/transformer_best.pth",
    "checkpoints/sparta_best.pth",
    "results/lstm_test_results.csv",
    "results/transformer_test_results.csv",
    "results/sparta_test_results.csv",
    "results/all_test_results.csv",
    "logs/lstm_train.log",
    "logs/transformer_train.log",
    "logs/sparta_train.log",
]


REQUIRED_OUTPUTS_GENERATE = [
    "raw_logs/node_log.csv",
    "raw_logs/link_log.csv",
    "raw_logs/service_log.csv",
    "raw_logs/path_log.csv",
    "raw_logs/sla_log.csv",
    "processed/path_graphs.pkl",
    "processed/labeled_samples.pkl",
    "processed/attributed_samples.pkl",
    "dataset/train.pkl",
    "dataset/val.pkl",
    "dataset/test.pkl",
    "audit/label_distribution.csv",
    "audit/scenario_distribution.csv",
    "audit/label_by_scenario.csv",
    "audit/attribution_distribution.csv",
    "audit/attribution_majority_baseline.csv",
    "audit/split_shift_report.csv",
    "audit/audit_report.txt",
]


REQUIRED_OUTPUTS_EVAL = [
    "results/lstm_test_results.csv",
    "results/transformer_test_results.csv",
    "results/sparta_test_results.csv",
    "results/all_test_results.csv",
    "results/lstm_confusion_matrix.csv",
    "results/transformer_confusion_matrix.csv",
    "results/sparta_confusion_matrix.csv",
    "results/lstm_prediction_distribution.csv",
    "results/transformer_prediction_distribution.csv",
    "results/sparta_prediction_distribution.csv",
    "results/lstm_per_class_recall.csv",
    "results/transformer_per_class_recall.csv",
    "results/sparta_per_class_recall.csv",
    "results/lstm_per_class_f1.csv",
    "results/transformer_per_class_f1.csv",
    "results/sparta_per_class_f1.csv",
    "logs/lstm_eval.log",
    "logs/transformer_eval.log",
    "logs/sparta_eval.log",
]


def check_outputs_or_report(run_dir: str | Path, required_files: list[str]) -> dict[str, list[str]]:
    result = check_required_outputs(Path(run_dir), required_files)
    if result["missing_outputs"]:
        print("[OUTPUT CHECK FAILED]")
        print("missing:")
        for rel in result["missing_outputs"]:
            print(f"- {rel}")
    else:
        print("[OUTPUT CHECK] all required outputs exist.")
    return result

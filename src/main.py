"""
Unified CLI — Visual Prompt Injection Detection System.

Usage
-----
  python src/main.py --train                     # Run full pipeline
  python src/main.py --predict path/to/image.jpg # Classify a single image
  python src/main.py --evaluate                  # Re-evaluate on test sets
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)  # suppress verbose lib output in CLI

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI Entry Points
# ---------------------------------------------------------------------------

def cmd_train():
    """Execute the full end-to-end training pipeline."""
    console.rule("[bold blue]Visual Prompt Injection Detection — Training Pipeline[/]")

    # Step 1: Data Collection (if needed)
    console.print("\n[bold]Step 1/5[/] — Data Collection & Verification …")
    _ensure_dataset()

    # Step 2: Preprocessing
    console.print("[bold]Step 2/5[/] — CV Preprocessing, OCR Extraction & Dataset Splits …")
    from src.preprocessing import run_preprocessing  # noqa: PLC0415
    train_df, test_df = run_preprocessing(verbose=False)
    console.print(
        f"  ✔ train.csv ([cyan]{len(train_df)}[/] rows)  "
        f"test.csv ([cyan]{len(test_df)}[/] rows)"
    )

    # Step 3: Supervised Learning
    console.print("[bold]Step 3/5[/] — Supervised Model Benchmark (SVM vs. Random Forest) …")
    from src.supervised_model import train_and_evaluate  # noqa: PLC0415
    metrics = train_and_evaluate(verbose=False)
    best = metrics.get("best_model", "N/A")
    best_m = metrics.get(best, {})
    console.print(
        f"  ✔ Best model: [green]{best}[/]  "
        f"F1=[cyan]{best_m.get('f1', 0):.4f}[/]  "
        f"Recall=[cyan]{best_m.get('recall', 0):.4f}[/]  "
        f"AUC=[cyan]{best_m.get('roc_auc', 0):.4f}[/]"
    )

    # Step 4: Unsupervised Clustering
    console.print("[bold]Step 4/5[/] — K-Means Clustering (k=3) + 2D PCA Visualisation …")
    from src.unsupervised_model import train_kmeans  # noqa: PLC0415
    train_kmeans(verbose=False)
    console.print("  ✔ kmeans_model.joblib  kmeans_clusters_pca.png")

    # Step 5: Figures & Reports
    console.print("[bold]Step 5/5[/] — Generating Figures & Evaluation Reports …")
    from src.evaluation import generate_all_figures, generate_final_summary  # noqa: PLC0415
    generate_all_figures(metrics, verbose=False)
    generate_final_summary(metrics, verbose=False)
    console.print("  ✔ confusion_matrix.png  roc_curve.png  top_malicious_words.png")
    console.print("  ✔ evaluation_metrics.json  final_summary.md")

    console.print()
    console.rule("[bold green]Pipeline Complete ✓[/]")
    _print_metrics_table(metrics)


def cmd_predict(image_path: str):
    """Classify a single image and display detailed verdict."""
    from src.computer_vision import extract_text_from_image  # noqa: PLC0415
    from src.nlp import extract_lexical_cues  # noqa: PLC0415
    from src.supervised_model import load_model, predict_text  # noqa: PLC0415

    img = Path(image_path)
    if not img.exists():
        console.print(f"[bold red]Error:[/] Image not found: {image_path}")
        sys.exit(1)

    console.rule(f"[bold blue]Classifying: {img.name}[/]")

    # CV + OCR
    console.print("\n[dim]Running computer vision preprocessing and OCR …[/dim]")
    ocr_result = extract_text_from_image(str(img))
    extracted_text = ocr_result.get("extracted_text", "")
    is_camouflaged = ocr_result.get("is_camouflaged", False)
    ocr_source = ocr_result.get("source", "unknown")

    camouflage_str = (
        "[bold red]⚠  CAMOUFLAGE DETECTED[/]" if is_camouflaged
        else "[green]✔  Clean (No Camouflage)[/]"
    )

    # Load model
    model_path = PROJECT_ROOT / "models" / "best_supervised_model.joblib"
    vec_path = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
    if not model_path.exists():
        console.print("[bold red]Error:[/] Model not found. Run [cyan]--train[/] first.")
        sys.exit(1)

    model, vectorizer = load_model(model_path, vec_path)
    result = predict_text(extracted_text if extracted_text else "[EMPTY_IMAGE]", model, vectorizer)
    label = result["predicted_label"]
    confidence = result["confidence"]
    verdict = result["verdict"]

    # Lexical cues
    cues = extract_lexical_cues(extracted_text)

    # Display
    verdict_color = "bold red" if label == 1 else "bold green"
    verdict_icon = "🚨" if label == 1 else "✅"

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Field", style="bold", width=22)
    table.add_column("Value")

    table.add_row("Image", str(img.resolve()))
    table.add_row("OCR Source", ocr_source)
    table.add_row("Visual Camouflage", camouflage_str)
    table.add_row(
        "Extracted Text",
        (extracted_text[:120] + " …") if len(extracted_text) > 120 else (extracted_text or "[none]"),
    )
    table.add_row("Trigger Words", ", ".join(cues) if cues else "[none]")
    table.add_row(
        "Confidence",
        f"[cyan]{confidence * 100:.1f}%[/]",
    )
    table.add_row(
        "Verdict",
        f"[{verdict_color}]{verdict_icon}  {verdict}[/]",
    )

    console.print(table)

    # Final banner
    if label == 1:
        console.print(Panel(
            f"[bold red]⛔  VERDICT: MALICIOUS / INJECTION[/]\n"
            f"Confidence: {confidence * 100:.1f}%\n"
            f"Triggers: {', '.join(cues) or 'pattern-based'}",
            border_style="red",
        ))
    else:
        console.print(Panel(
            f"[bold green]✅  VERDICT: SAFE[/]\n"
            f"Confidence: {confidence * 100:.1f}%",
            border_style="green",
        ))


def cmd_evaluate():
    """Re-run evaluation on both test sets and print a metrics summary table."""
    import json  # noqa: PLC0415
    console.rule("[bold blue]Re-Evaluation — Visual Prompt Injection Detection[/]")

    metrics_path = PROJECT_ROOT / "results" / "reports" / "evaluation_metrics.json"
    if not metrics_path.exists():
        console.print("[yellow]Metrics file not found. Running evaluation now …[/yellow]")
        from src.supervised_model import train_and_evaluate  # noqa: PLC0415
        metrics = train_and_evaluate(verbose=False)
    else:
        with open(metrics_path) as f:
            metrics = json.load(f)

    _print_metrics_table(metrics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dataset():
    """Check the synthetic dataset exists; generate if missing."""
    benign_dir = PROJECT_ROOT / "data" / "raw" / "benign"
    malicious_dir = PROJECT_ROOT / "data" / "raw" / "malicious"
    metadata = PROJECT_ROOT / "data" / "raw" / "dataset_metadata.csv"

    benign_count = len(list(benign_dir.glob("*.jpg"))) if benign_dir.exists() else 0
    malicious_count = len(list(malicious_dir.glob("*.jpg"))) if malicious_dir.exists() else 0

    if benign_count >= 2500 and malicious_count >= 2500 and metadata.exists():
        console.print(f"  ✔ Dataset exists: [cyan]{benign_count}[/] benign, [cyan]{malicious_count}[/] malicious images")
        return

    console.print("  ⚙ Generating synthetic dataset …")
    from src.data_collection import generate_synthetic_dataset  # noqa: PLC0415
    generate_synthetic_dataset(count=5000)
    console.print("  ✔ Dataset generated.")


def _print_metrics_table(metrics: dict):
    """Print a formatted metrics summary using rich Table."""
    best = metrics.get("best_model", "N/A")
    console.print(f"\n[bold]Best Model:[/] [cyan]{best}[/]\n")

    for split_name, split_key in [
        ("In-Distribution Test (Synthetic)", best),
        ("Out-of-Distribution (CyberSecEval 3)", "cyberseceval3"),
    ]:
        m = metrics.get(split_key, {})
        if not m:
            continue
        table = Table(title=split_name, box=box.SIMPLE_HEAD)
        table.add_column("Metric", style="bold")
        table.add_column("Score", justify="right")
        for k, label in [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall ⬆"),
            ("f1", "F1-Score"),
            ("roc_auc", "ROC-AUC"),
        ]:
            val = m.get(k, None)
            if val is not None:
                table.add_row(label, f"[cyan]{val:.4f}[/]")
        console.print(table)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="visual_prompt_injection_detection",
        description="Visual Prompt Injection Detection System — CLI",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run the full training pipeline.")
    group.add_argument("--predict", metavar="IMAGE_PATH", help="Classify a single image.")
    group.add_argument("--evaluate", action="store_true", help="Re-evaluate and display metrics.")

    args = parser.parse_args()

    if args.train:
        cmd_train()
    elif args.predict:
        cmd_predict(args.predict)
    elif args.evaluate:
        cmd_evaluate()


if __name__ == "__main__":
    main()

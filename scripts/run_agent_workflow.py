"""Run the ClinGuard AI LangGraph workflow."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinguard.preprocessing.inspection import load_dataset
from clinguard.agents.workflow import run_clinguard_workflow


def main():

    input_path = (
        ROOT / "data" / "processed"
        / "clinguard_hybrid_detection.csv"
    )

    output_path = (
        ROOT / "data" / "processed"
        / "clinguard_agent_results.csv"
    )

    dataset = load_dataset(input_path)

    print("\nCLINGUARD AI - LANGGRAPH WORKFLOW")
    print("=" * 45)

    result = run_clinguard_workflow(
        dataset,
        strategy="or",
    )

    final_dataset = result["explained_dataset"]
    summary = result["summary"]

    final_dataset.to_csv(output_path, index=False)

    print("\nWORKFLOW SUMMARY")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nWorkflow completed successfully.")
    print("Output saved to:", output_path)


if __name__ == "__main__":
    main()
"""LangGraph workflow for ClinGuard AI."""

from typing import TypedDict

import pandas as pd
from langgraph.graph import StateGraph, START, END

from clinguard.verification.context_verification import (
    verify_anomaly_context,
)

from clinguard.explainability.explanations import (
    explain_anomaly_records,
)


class ClinGuardState(TypedDict):
    input_dataset: pd.DataFrame
    strategy: str
    verified_dataset: pd.DataFrame
    explained_dataset: pd.DataFrame
    summary: dict


def context_verification_node(state: ClinGuardState):

    verified = verify_anomaly_context(
        state["input_dataset"],
        strategy=state["strategy"],
    )

    return {"verified_dataset": verified}


def explainability_node(state: ClinGuardState):

    explained = explain_anomaly_records(
        state["verified_dataset"]
    )

    return {"explained_dataset": explained}


def summary_node(state: ClinGuardState):

    dataset = state["explained_dataset"]

    summary = {
        "total_records": len(dataset),

        "supported_candidates": int(
            (
                dataset["verification_status"]
                == "supported_anomaly_candidate"
            ).sum()
        ),

        "candidates_needing_review": int(
            (
                dataset["verification_status"]
                == "candidate_needs_review"
            ).sum()
        ),

        "quality_reviews_without_model_alert": int(
            (
                dataset["verification_status"]
                == "quality_review_without_model_alert"
            ).sum()
        ),

        "no_model_alert": int(
            (
                dataset["verification_status"]
                == "no_model_alert"
            ).sum()
        ),
    }

    return {"summary": summary}


def build_clinguard_workflow():

    graph = StateGraph(ClinGuardState)

    graph.add_node(
        "context_verification",
        context_verification_node,
    )

    graph.add_node(
        "explainability",
        explainability_node,
    )

    graph.add_node(
        "summary",
        summary_node,
    )

    graph.add_edge(START, "context_verification")
    graph.add_edge("context_verification", "explainability")
    graph.add_edge("explainability", "summary")
    graph.add_edge("summary", END)

    return graph.compile()


def run_clinguard_workflow(dataset, strategy="or"):

    workflow = build_clinguard_workflow()

    initial_state = {
        "input_dataset": dataset,
        "strategy": strategy,
    }

    return workflow.invoke(initial_state)
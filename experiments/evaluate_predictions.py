import os
from dotenv import load_dotenv
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain_openai import ChatOpenAI
import argparse
from langsmith import Client


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- CONFIG ---
DATASET_NAME = "State of AI Across the Years!"
PROJECT_NAME = "State of AI Across the Years!"
EVAL_LLM_MODEL = "gpt-4.1"  # Match the notebook's model if possible

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_tag", type=str, help="Only evaluate runs with this experiment_tag")
args = parser.parse_args()

if args.experiment_tag:
    print(f"Evaluating only runs with experiment_tag: {args.experiment_tag}")

client = Client()
runs = list(client.list_runs(
    project_name=PROJECT_NAME,
    dataset_name=DATASET_NAME,
    filters={"metadata.experiment_tag": args.experiment_tag} if args.experiment_tag else None,
))

# --- EVALUATORS ---
eval_llm = ChatOpenAI(model=EVAL_LLM_MODEL)

qa_evaluator = LangChainStringEvaluator("qa", config={"llm": eval_llm})

labeled_helpfulness_evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "helpfulness": (
                "Is this submission helpful to the user,"
                " taking into account the correct reference answer?"
            )
        },
        "llm": eval_llm
    },
    prepare_data=lambda run, example: {
        "prediction": run.outputs["output"],
        "reference": example.outputs["answer"],
        "input": example.inputs["question"],
    }
)

dope_or_nope_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "dopeness": "Is this submission dope, lit, or cool?",
        },
        "llm": eval_llm
    }
)

# --- RUN EVALUATION ---
if __name__ == "__main__":
    print("Running evaluation on predictions in LangSmith...")
    results = evaluate(
        runs,
        data=DATASET_NAME,
        evaluators=[
            qa_evaluator,
            labeled_helpfulness_evaluator,
            dope_or_nope_evaluator
        ],
        project_name=PROJECT_NAME,
        metadata={"source": "app_evaluation"},
    )
    print("Evaluation complete! View results in your LangSmith dashboard.") 
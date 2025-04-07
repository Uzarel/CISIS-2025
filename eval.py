import pandas as pd
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerAccuracy
from ragas.llms import LangchainLLMWrapper
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

# Setup evaluator
evaluator_llm = LangchainLLMWrapper(OllamaLLM(model="llama3.3", temperature=0))
answer_accuracy = AnswerAccuracy(llm=evaluator_llm)

# Define helper to evaluate one strategy
def evaluate_strategy(df, strategy: str):
    results = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Evaluating {strategy}"):
        sample = SingleTurnSample(
            user_input=row.question,
            response=getattr(row, f'{strategy}_answer'),
            reference=row.answer
        )
        score = answer_accuracy.single_turn_score(sample)
        results.append(score)
    return results

for split in ['inheritance', 'divorce']:
    # Load data
    answers_df = pd.read_csv(f"evaluation/{split}_qa.csv")
    gt_df = pd.read_csv(f"evaluation/{split}_testset.csv")

    # Merge on question
    df = pd.merge(answers_df, gt_df[['question', 'answer']], on='question', how='inner')
    
    df['multiquery_accuracy'] = evaluate_strategy(df, 'multiquery')
    df['decomposition_accuracy'] = evaluate_strategy(df, 'decomposition')
    df['stepback_accuracy'] = evaluate_strategy(df, 'stepback')
    df.to_csv(f"evaluation/{split}_ragas_accuracy_scores.csv", index=False)
    print(f"Scores saved to {split}_ragas_accuracy_scores.csv")

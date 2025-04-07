import pandas as pd
import time
from pipelines.multiquery import run_multiquery_pipeline
from pipelines.decomposition import run_decomposition_pipeline
from pipelines.stepback import run_stepback_pipeline

def process_questions(csv_file: str, output_file: str):
    # Load CSV file with column 'question'
    df = pd.read_csv(csv_file)
    
    total_questions = len(df)
    results = []
    
    for idx, row in df.iterrows():
        question = row['question']
        remaining = total_questions - idx - 1
        print(f"\nProcessing question {idx+1} of {total_questions} (Remaining: {remaining}): {question}")
        
        # Multiquery pipeline
        start_time = time.time()
        answer_multi = run_multiquery_pipeline(question)
        multi_time = time.time() - start_time
        print("Multiquery Answer:")
        print(answer_multi)
        
        # Decomposition pipeline
        start_time = time.time()
        answer_decomp = run_decomposition_pipeline(question)
        decomp_time = time.time() - start_time
        print("Decomposition Answer:")
        print(answer_decomp)
        
        # Stepback pipeline
        start_time = time.time()
        answer_stepback = run_stepback_pipeline(question)
        stepback_time = time.time() - start_time
        print("Stepback Answer:")
        print(answer_stepback)
        
        results.append({
            "question": question,
            "multiquery_answer": answer_multi,
            "multiquery_time": multi_time,
            "decomposition_answer": answer_decomp,
            "decomposition_time": decomp_time,
            "stepback_answer": answer_stepback,
            "stepback_time": stepback_time
        })
        
        # Save progress after each iteration
        pd.DataFrame(results).to_csv(output_file, index=False)
    
    return results

if __name__ == "__main__":
    for split in ["inheritance", "divorce"]:
        csv_path = f"evaluation/{split}_testset.csv"
        output_file = f"evaluation/{split}_qa.csv"
        process_questions(csv_path, output_file)

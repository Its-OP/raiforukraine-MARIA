import os
import json
import pandas as pd

import nltk
import ssl
# disable SSL check to download nltk pakages on MacOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from src.eval.eval_dataframe import eval_dataframe_parallel
from dotenv import load_dotenv

def main():
    load_dotenv()
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

    with open('src/eval/prompts.json', 'r') as file:
        # Load the JSON data from the file
        system_prompts = json.load(file)

    df_mcq = pd.read_csv(os.environ.get('MODEL_MCQ_PATH'))
    df_lisa_sheets = pd.read_csv(os.environ.get('LISA_SHEETS_PATH'))
    
    # test small subset
    # common_ids = df_mcq['id'].isin(df_lisa_sheets['id'])
    # df_mcq = df_mcq[common_ids].iloc[:60]
    
    df_lisa_sheets = df_lisa_sheets[df_lisa_sheets['id'].isin(df_mcq['id'])]

    df_eval = eval_dataframe_parallel(df_mcqs=df_mcq,
                                      df_lisa_sheets=df_lisa_sheets,
                                      openai_key=OPENAI_KEY,
                                      num_workers=10,
                                      lisa_sheet_id_col='id',
                                      lisa_sheet_col='content_gpt',
                                      compute_answerability=False,
                                      compute_originality=False,
                                      compute_readability=False,
                                      compute_negation=False,
                                      compute_is_question=False,
                                      compute_relevance=False,
                                      compute_ambiguity=False,
                                      compute_disclosure=False,
                                      compute_difficulty=False,
                                      compute_distractors_quality=True,
                                      distractors_quality_system_prompt=system_prompts['distractors_quality_prompt'],
                                      distractors_quality_col='distractor_quality',
                                      merge=False # set to True if your dataframe does not have the Lisa Sheet content
                                      )

    df_eval.to_csv(os.environ.get('MODEL_MCQ_EVAL_EXPORT_PATH'), index=False)

if __name__ == '__main__':
    main()

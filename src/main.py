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
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    with open('/home/adam/Conda/lig/raiforukraine-MARIA/src/eval/prompts.json', 'r') as file:
        system_prompts = json.load(file)
    df_mcq = pd.read_csv(os.environ.get('MODEL_MCQ_PATH'))
    df_lisa_sheets = pd.read_csv(os.environ.get('LISA_SHEETS_PATH'))
    # test small subset
    common_ids = df_mcq['id'].isin(df_lisa_sheets['id'])
    #df_mcq = df_mcq[common_ids].iloc[:2000]
    df_mcq = df_mcq[common_ids]
    df_lisa_sheets = df_lisa_sheets[df_lisa_sheets['id'].isin(df_mcq['id'])]
    df_eval = eval_dataframe_parallel(df_mcqs=df_mcq,
                                        df_lisa_sheets=df_lisa_sheets,
                                        openai_key=OPENAI_API_KEY,
                                        num_workers=20,
                                        lisa_sheet_id_col='id',
                                        lisa_sheet_col='content_gpt',
                                        answerability_system_prompt=system_prompts['answerability_prompt'],
                                        disclosure_system_prompt=system_prompts['disclosure_prompt'],
                                        difficulty_system_prompt=system_prompts['difficulty_prompt'],
                                        )
    df_eval.to_csv(os.environ.get('MODEL_MCQ_EVAL_EXPORT_PATH'), index=False)

if __name__ == '__main__':
    main()

import json

import pandas as pd
from openai import OpenAI

from src.eval.llm_evaluation import generate_prompt_for_question, call_openai_api


def compute_disclosure_for_df(df: pd.DataFrame,
                              api_key: str,
                              question_col: str,
                              disclosure_col: str,
                              system_prompt: str,
                              temp: float,
                              max_completion_tokens: int):

    client = create_client(api_key)  # Use the new function            
    def disclosure_applicable(row):
        user_prompt = generate_prompt_for_question(row,
                                                   question_col=question_col,
                                                   include_options=False)
        
        return call_openai_api(client, system_prompt, user_prompt, temp=temp, max_completion_tokens=max_completion_tokens)

    df[disclosure_col] = df.apply(disclosure_applicable, axis=1)
    return df
    

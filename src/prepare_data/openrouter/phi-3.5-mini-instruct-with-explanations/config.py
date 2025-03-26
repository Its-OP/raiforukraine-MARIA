import os
from dotenv import load_dotenv

load_dotenv()

# Changed from OPENAI_KEY to TOGETHER_API_KEY
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

DEFAULT_PROMPT = """Based on the following educational content, generate a
multiple-choice question with four answer options where only one is correct.
The question and its options must adhere to the following rule:

1. **Ambiguity Between Correct and Incorrect Options**: The incorrect options
(distractors) should be plausible and logically related to the question,
creating ambiguity for someone who may not have complete knowledge of the
topic. Distractors should reflect common misconceptions or misunderstandings
that could reasonably confuse the respondent.
"""

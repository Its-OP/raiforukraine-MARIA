import os
import json
import pandas as pd
import torch
import torch.nn.functional as F
import re
import logging
import time
from datetime import datetime, timedelta
from collections import deque
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Configuration for API calls
API_CONFIG = {
    'temperature': 0.7,  # Centralized temperature setting
    'max_tokens': 1,
    'rpm_limit': 300,
    'batch_size': 50
}

# For NLTK
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ==== RATE LIMITER ====

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def wait_if_needed(self):
        now = datetime.now()
        while self.requests and self.requests[0] < now - timedelta(seconds=self.time_window):
            self.requests.popleft()
        if len(self.requests) >= self.max_requests:
            wait_time = (self.requests[0] + timedelta(seconds=self.time_window) - now).total_seconds()
            if wait_time > 0:
                time.sleep(wait_time)
            self.requests.popleft()
        self.requests.append(now)

# ==== UTILITY FUNCTIONS ====

def load_model(model_name="BAAI/bge-base-en-v1.5"):
    """Loads the model and tokenizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return model, tokenizer, device

def generate_embedding(text, model, tokenizer, device):
    """Generates an embedding for the given text"""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0)

def create_client(api_key):
    """Create and return an API client for LLM calls"""
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1"
    )

def create_rate_limited_client(api_key):
    """Create API client with rate limiting"""
    client = create_client(api_key)
    rate_limiter = RateLimiter(max_requests=API_CONFIG['rpm_limit'], time_window=60)
    return client, rate_limiter

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_openai_api_with_rate_limit(client, rate_limiter, system_prompt, user_prompt):
    """Call the LLM API with rate limiting and retry logic"""
    try:
        rate_limiter.wait_if_needed()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=API_CONFIG['temperature'],
            max_tokens=API_CONFIG['max_tokens'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def generate_prompt_for_question(row,
                               question_col='question',
                               option_a_col='option_a',
                               option_b_col='option_b',
                               option_c_col='option_c',
                               option_d_col='option_d',
                               correct_option='correct_option',
                               include_options=True,
                               include_correct_option=False,
                               context_col=None):
    """Generate a prompt for the question with appropriate formatting"""
    question_text = row[question_col]
    options = f"a) {row[option_a_col]}\nb) {row[option_b_col]}\nc) {row[option_c_col]}\nd) {row[option_d_col]}"
    correct_option_text = row[correct_option]
    
    user_prompt_delimiter = "-----\n"
    user_prompt_question = f"Question:\n{question_text}\n"
    user_prompt_options = f"Options:\n{options}\n"

    user_prompt = user_prompt_delimiter + user_prompt_question
    if include_options:
        user_prompt += user_prompt_options
    if include_correct_option:
        correct_option_info = f"Correct option: {correct_option_text}\n"
        user_prompt += correct_option_info

    user_prompt += user_prompt_delimiter
    
    if context_col is not None and context_col in row:
        mcq_context = f"""Context:\n-----\n{row[context_col]}\n-----\n"""
        user_prompt = mcq_context + user_prompt

    return user_prompt

# ==== EVALUATION MODULES ====

def compute_ambiguity(df, ambiguity_col, correct_option_col, option_a_col, option_b_col, option_c_col, option_d_col):
    """Calculate ambiguity between correct and incorrect options"""
    def calculate_ambiguity(row):
        try:
            correct_option = str(row[correct_option_col]).lower() if row[correct_option_col] is not None else ""
            options = {
                'a': str(row[option_a_col]) if row[option_a_col] is not None else "",
                'b': str(row[option_b_col]) if row[option_b_col] is not None else "",
                'c': str(row[option_c_col]) if row[option_c_col] is not None else "",
                'd': str(row[option_d_col]) if row[option_d_col] is not None else ""
            }
            
            if not correct_option or correct_option not in options:
                return 0.5
            
            correct_text = options[correct_option]
            length_diffs = [abs(len(correct_text) - len(opt)) for opt in options.values()]
            avg_length_diff = sum(length_diffs) / len(length_diffs)
            
            normalized_ambiguity = 1.0 / (1.0 + avg_length_diff/10)
            return normalized_ambiguity
            
        except Exception as e:
            print(f"Error in ambiguity calculation: {e}")
            return 0.5
    
    tqdm.pandas(desc="Computing Ambiguity")
    df[ambiguity_col] = df.progress_apply(calculate_ambiguity, axis=1)
    return df

def compute_answerability_with_rate_limit(df, api_key, question_col, option_cols, context_col, 
                                        model_answer_col, system_prompt):
    """Compute answerability with rate limiting"""
    client, rate_limiter = create_rate_limited_client(api_key)
    results = []
    
    for i in range(0, len(df), API_CONFIG['batch_size']):
        batch = df.iloc[i:i + API_CONFIG['batch_size']]
        batch_results = []
        
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Answerability Batch {i//API_CONFIG['batch_size'] + 1}"):
            user_prompt = generate_prompt_for_question(
                row,
                question_col=question_col,
                option_a_col=option_cols[0],
                option_b_col=option_cols[1],
                option_c_col=option_cols[2],
                option_d_col=option_cols[3],
                context_col=context_col
            )
            
            result = call_openai_api_with_rate_limit(client, rate_limiter, system_prompt, user_prompt)
            batch_results.append(result)
            
        results.extend(batch_results)
    
    df[model_answer_col] = results
    return df

def compute_disclosure_with_rate_limit(df, api_key, question_col, disclosure_col, system_prompt):
    """Check for disclosure hints with rate limiting"""
    client, rate_limiter = create_rate_limited_client(api_key)
    results = []
    
    for i in range(0, len(df), API_CONFIG['batch_size']):
        batch = df.iloc[i:i + API_CONFIG['batch_size']]
        batch_results = []
        
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Disclosure Batch {i//API_CONFIG['batch_size'] + 1}"):
            user_prompt = generate_prompt_for_question(
                row,
                question_col=question_col,
                include_options=False
            )
            
            result = call_openai_api_with_rate_limit(client, rate_limiter, system_prompt, user_prompt)
            batch_results.append(result)
            
        results.extend(batch_results)
    
    df[disclosure_col] = results
    return df

def compute_difficulty_with_rate_limit(df, api_key, question_col, difficulty_col, system_prompt):
    """Evaluate difficulty with rate limiting"""
    client, rate_limiter = create_rate_limited_client(api_key)
    results = []
    
    for i in range(0, len(df), API_CONFIG['batch_size']):
        batch = df.iloc[i:i + API_CONFIG['batch_size']]
        batch_results = []
        
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Difficulty Batch {i//API_CONFIG['batch_size'] + 1}"):
            user_prompt = generate_prompt_for_question(
                row,
                question_col=question_col,
                include_options=True,
                include_correct_option=True
            )
            
            result = call_openai_api_with_rate_limit(client, rate_limiter, system_prompt, user_prompt)
            batch_results.append(result)
            
        results.extend(batch_results)
    
    df[difficulty_col] = results
    return df

def calculate_relevance(df, relevance_col, question_col, context_col, model_name="BAAI/bge-base-en-v1.5"):
    """Calculate semantic relevance between questions and context"""
    model, tokenizer, device = load_model(model_name)

    def relevance_score(row):
        q_text = str(row[question_col]) if row[question_col] is not None else ""
        ctx_text = str(row[context_col]) if row[context_col] is not None else ""
        
        q_emb = generate_embedding(q_text, model, tokenizer, device)
        l_emb = generate_embedding(ctx_text, model, tokenizer, device)
        similarity = F.cosine_similarity(q_emb, l_emb, dim=0).item()
        return similarity

    tqdm.pandas(desc="Computing Relevance")
    df[relevance_col] = df.progress_apply(relevance_score, axis=1)
    return df

def calculate_readability(df, readability_col, question_col):
    """Calculate readability score using Flesch-Kincaid"""
    def syllable_count(word):
        word = word.lower()
        syllable_count = len(re.findall(r'[aeiouy]+', word))
        return max(1, syllable_count)

    def compute_readability(text):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        words = text.split()
        num_words = len(words)
        num_sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        num_syllables = sum(syllable_count(word) for word in words)

        if num_words == 0:
            return 0
            
        readability = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        return readability

    tqdm.pandas(desc="Computing Readability")
    df[readability_col] = df.progress_apply(lambda row: compute_readability(row[question_col]), axis=1)
    return df

def calculate_originality(df, originality_col, question_col, context_col):
    """Calculate originality of questions compared to context"""
    def compute_originality(row):
        q_text = str(row[question_col]) if row[question_col] is not None else ""
        ctx_text = str(row[context_col]) if row[context_col] is not None else ""
        return len(q_text) / max(len(ctx_text), 1)
    
    tqdm.pandas(desc="Computing Originality")
    df[originality_col] = df.progress_apply(compute_originality, axis=1)
    return df

def check_negation(df, negation_col, question_col):
    """Check if questions start with negative words"""
    def starts_with_negation(text):
        if not isinstance(text, str) or text is None:
            text = str(text) if text is not None else ""
            
        negation_patterns = ["not", "no ", "isn't", "aren't", "wasn't", "weren't",
                            "don't", "doesn't", "didn't", "can't", "couldn't", 
                            "won't", "wouldn't", "shouldn't", "isn't", "aren't"]
        
        text_lower = text.lower()
        words = text_lower.split()
        if len(words) >= 1:
            return any(pattern in text_lower[:50] for pattern in negation_patterns)
        return False

    tqdm.pandas(desc="Checking Negation")
    df[negation_col] = df.progress_apply(lambda row: starts_with_negation(row[question_col]), axis=1)
    return df

def check_is_question(df, is_question_col, question_col):
    """Verify if text is formatted as a question"""
    def is_question(text):
        if not isinstance(text, str) or text is None:
            text = str(text) if text is not None else ""
            
        if text.strip().endswith("?"):
            return True
        
        question_words = r"^(who|what|where|when|why|how|does|should|do|did|could|will|would)\b"

        if re.match(question_words, text.strip(), re.IGNORECASE):
            return True
        
        return False

    tqdm.pandas(desc="Checking Question Format")
    df[is_question_col] = df.progress_apply(lambda row: is_question(row[question_col]), axis=1)
    return df

# ==== MAIN EVALUATION FUNCTION ====

def evaluate_mcq_dataset(df_mcq, df_context, id_col='id', context_col='content_gpt', 
                        output_path='./mcqs_eval.csv', api_key=None, system_prompts=None):
    """Main function to evaluate MCQ dataset with rate limiting"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Merging MCQ and context dataframes")
    df_merged = pd.merge(df_mcq, df_context[[id_col, context_col]], on=id_col, how='left')
    logger.info(f"Total rows after merge: {len(df_merged)}")
    
    # Non-API evaluations
    try:
        logger.info("Calculating originality")
        df_merged = calculate_originality(df_merged, 'originality', 'question', context_col)
        df_merged.to_csv(f"{output_path}.originality.tmp", index=False)
    except Exception as e:
        logger.error(f"Error in originality calculation: {e}")
    
    try:
        logger.info("Calculating readability")
        df_merged = calculate_readability(df_merged, 'readability', 'question')
        df_merged.to_csv(f"{output_path}.readability.tmp", index=False)
    except Exception as e:
        logger.error(f"Error in readability calculation: {e}")
    
    try:
        logger.info("Checking for negation")
        df_merged = check_negation(df_merged, 'starts_with_negation', 'question')
        df_merged.to_csv(f"{output_path}.negation.tmp", index=False)
    except Exception as e:
        logger.error(f"Error in negation check: {e}")
    
    try:
        logger.info("Checking question format")
        df_merged = check_is_question(df_merged, 'is_question', 'question')
        df_merged.to_csv(f"{output_path}.is_question.tmp", index=False)
    except Exception as e:
        logger.error(f"Error in question format check: {e}")
    
    try:
        logger.info("Calculating relevance")
        df_merged = calculate_relevance(df_merged, 'relevance', 'question', context_col)
        df_merged.to_csv(f"{output_path}.relevance.tmp", index=False)
    except Exception as e:
        logger.error(f"Error in relevance calculation: {e}")
    
    try:
        logger.info("Calculating ambiguity")
        df_merged = compute_ambiguity(
            df_merged, 'ambiguity', 'correct_option', 
            'option_a', 'option_b', 'option_c', 'option_d')
        df_merged.to_csv(f"{output_path}.ambiguity.tmp", index=False)
    except Exception as e:
        logger.error(f"Error in ambiguity calculation: {e}")
    
    # API-dependent evaluations
    if api_key and system_prompts:
        if 'answerability_prompt' in system_prompts:
            try:
                logger.info("Running answerability evaluation with rate limiting")
                df_merged = compute_answerability_with_rate_limit(
                    df_merged, api_key, 'question', 
                    ['option_a', 'option_b', 'option_c', 'option_d'],
                    context_col, 'gpt_answer', 
                    system_prompts['answerability_prompt']
                )
                df_merged.to_csv(f"{output_path}.answerability.tmp", index=False)
            except Exception as e:
                logger.error(f"Error in answerability evaluation: {e}")
        
        if 'disclosure_prompt' in system_prompts:
            try:
                logger.info("Running disclosure evaluation with rate limiting")
                df_merged = compute_disclosure_with_rate_limit(
                    df_merged, api_key, 'question', 'disclosure',
                    system_prompts['disclosure_prompt']
                )
                df_merged.to_csv(f"{output_path}.disclosure.tmp", index=False)
            except Exception as e:
                logger.error(f"Error in disclosure evaluation: {e}")
        
        if 'difficulty_prompt' in system_prompts:
            try:
                logger.info("Running difficulty evaluation with rate limiting")
                df_merged = compute_difficulty_with_rate_limit(
                    df_merged, api_key, 'question', 'difficulty',
                    system_prompts['difficulty_prompt']
                )
                df_merged.to_csv(f"{output_path}.difficulty.tmp", index=False)
            except Exception as e:
                logger.error(f"Error in difficulty evaluation: {e}")
    
    logger.info("Saving final results")
    df_merged.to_csv(output_path, index=False)
    return df_merged

# ==== MAIN EXECUTION ====

def main():
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    # System prompts
    system_prompts = {
        "difficulty_prompt": """You are tasked with evaluating the ambiguity between the correct option and the incorrect options for a multiple-choice item intended for use in a medical institution exam. Your job is to assess this ambiguity on a scale of 1 to 5. Specifically, focus on how similar or distinct the correct option is relative to the incorrect options.

Use the following guidelines for scoring:

- Score 1 – Excellent Ambiguity: The correct option is nearly indistinguishable from the incorrect options; all options are highly plausible, logically connected, and reflective of common misconceptions.
- Score 2 – High Ambiguity: The correct option is quite similar to the incorrect options, making differentiation challenging, though subtle distinctions exist.
- Score 3 – Moderate Ambiguity: The correct option is somewhat distinct yet still shares noticeable similarities with the incorrect options, presenting a moderate challenge.
- Score 4 – Minimal Ambiguity: The correct option is clearly more distinct than the incorrect options, though one or two may share a slight resemblance.
- Score 5 – No Meaningful Ambiguity: The correct option is obviously different from the incorrect options, which appear irrelevant, overly obvious, or unrelated.

After reviewing the options, provide only a numerical score from 1 to 5 that best represents the level of ambiguity.""",

        "disclosure_prompt": """You are tasked with evaluating a multiple-choice question (which will be provided after this prompt) intended for use in a medical institution exam.
Determine if the way the question is constructed would allow a test taker with no relevant medical knowledge to identify the correct answer through clues in the phrasing, structure, answer choice formatting, or other linguistic hints.
If there are any such clues that would help an uninformed test taker guess the correct answer, respond with "True".
If not, respond with "False".
Provide no additional text besides either "True" or "False".""",

        "answerability_prompt": """You are tasked with answering multiple-choice questions, containing 4 different answer options - a, b, c and d.
You are given some context to help you answer the question.
Provide just a single letter corresponding to the correct option as the response."""
    }
    
    # Load data
    df_mcq = pd.read_csv(os.environ.get('MODEL_MCQ_PATH'))
    df_lisa_sheets = pd.read_csv(os.environ.get('LISA_SHEETS_PATH'))
    
    # Filter to common IDs
    common_ids = df_mcq['id'].isin(df_lisa_sheets['id'])
    #df_mcq = df_mcq[common_ids].iloc[:10]
    df_mcq = df_mcq[common_ids]
    df_lisa_sheets = df_lisa_sheets[df_lisa_sheets['id'].isin(df_mcq['id'])]
    
    # Run evaluation
    results = evaluate_mcq_dataset(
        df_mcq=df_mcq,
        df_context=df_lisa_sheets,
        id_col='id',
        context_col='content_gpt',
        output_path=os.environ.get('MODEL_MCQ_EVAL_EXPORT_PATH'),
        api_key=OPENAI_API_KEY,
        system_prompts=system_prompts
    )
    
    print(f"Evaluation complete. Results saved to {os.environ.get('MODEL_MCQ_EVAL_EXPORT_PATH')}")

if __name__ == '__main__':
    main()        

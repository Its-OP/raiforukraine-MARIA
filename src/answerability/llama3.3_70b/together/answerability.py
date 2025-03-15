import json
import argparse
import sys
import time
import os
import re
from openai import OpenAI
from tqdm import tqdm  # For progress bar

parser = argparse.ArgumentParser()
parser.add_argument('-l', "--llm", type=str, default='Mistral')
parser.add_argument('-t', "--temperature", type=float, default=0.5)
args = parser.parse_args()
input_file = 'mcqs_answerability.json'
output_file = 'answerability'

# Define models for Together API
model_name = None
if args.llm == 'Mistral':
    output_file += f'_mistral_16_{args.temperature}.json'
    model = 'mistral-small:24b-instruct-2501-fp16'
elif args.llm == 'Instruct':
    output_file += f'_mistral_8_{args.temperature}.json'
    model = 'mistral-small:24b-instruct-2501-q8_0'
elif args.llm == 'Llama8':
    model = 'llama3.1:8b-instruct-q8_0'
    output_file += f'_Llama_8_{args.temperature}.json'
elif args.llm == 'Llama16':
    model = 'llama3.1:8b-instruct-fp16'
    output_file += f'_Llama_16_{args.temperature}.json'
elif args.llm == 'Llama3.3_70b':
    model_name = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
    output_file += f'_Llama_3.3_70b_{args.temperature}.json'
elif args.llm == 'Llama3.1_405b':
    model_name = 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'
    output_file += f'_Llama_3.1_405b_{args.temperature}.json'    
else:
    sys.exit('ERROR Dataset !!!')

# Initialize Together API client
client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

# Load MCQs from input file
with open(input_file, 'r') as file:
    mcqs = json.load(file)

#mcqs = mcqs[:100]
    
results = []
excepts = []

# Function to extract JSON from response
def extract_json_answer(response_text):
    """Extract answer from response, handling potential JSON parsing issues"""
    if not response_text:
        return "ERROR"
    
    # Try to parse as JSON first
    try:
        # Check if the entire text is valid JSON
        answer_obj = json.loads(response_text)
        if 'answer' in answer_obj:
            answer = answer_obj['answer'].strip().upper()
            if answer in ['A', 'B', 'C', 'D']:
                return answer
    except json.JSONDecodeError:
        # If not valid JSON, try to extract JSON-like content
        if '{' in response_text and '}' in response_text:
            try:
                # Extract text between first { and last }
                json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
                answer_obj = json.loads(json_part)
                if 'answer' in answer_obj:
                    answer = answer_obj['answer'].strip().upper()
                    if answer in ['A', 'B', 'C', 'D']:
                        return answer
            except:
                pass
    
    # If JSON parsing fails, try regex patterns
    patterns = [
        r'"answer"\s*:\s*"([A-D])"',     # "answer": "A"
        r'"answer"\s*:\s*([A-D])',       # "answer": A
        r"'answer'\s*:\s*'([A-D])'",     # 'answer': 'A'
        r'answer\s*:\s*"([A-D])"',       # answer: "A"
        r'answer\s*:\s*([A-D])',         # answer: A
        r'([A-D])\s*is the answer',      # A is the answer
        r'The answer is\s+([A-D])',      # The answer is A
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Last resort: check if the response is just a letter
    clean_response = response_text.strip().upper()
    if clean_response in ['A', 'B', 'C', 'D']:
        return clean_response
    
    # If all else fails, return error
    return f"PARSE_ERROR: {response_text[:30]}..."

# Template for prompting - Using JSON format
template = """Answer the following {subject} multiple-choice question:

Question: {question}
Options:
{options}

Return your answer as a JSON object with a single field 'answer' containing the letter (A, B, C, or D) corresponding to the correct answer.
Example: {{"answer": "A"}}

Only provide the JSON object, nothing else."""

# Process each MCQ with progress bar
for idx, mcq in tqdm(enumerate(mcqs), total=len(mcqs), desc=f"Processing {args.llm} questions"):
    question = mcq['question']
    options = f"A: {mcq['opa']}\nB: {mcq['opb']}\nC: {mcq['opc']}\nD: {mcq['opd']}"
    correct_answer = mcq['cop'].upper()  # Ensure uppercase for comparison
    
    # Format the prompt for this question
    prompt_content = template.format(
        subject=mcq['subject_name'].lower(),
        question=question,
        options=options
    )
    
    # Start timing
    beg = time.time()
    
    try:
        # Make API request to Together
        # Note: NOT using response_format parameter which isn't supported
        response = client.chat.completions.create(
            model=model_name or model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions with JSON objects."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=args.temperature
        )
        
        # Extract the response
        llm_response = response.choices[0].message.content
        
        # Extract JSON answer
        llm_answer = extract_json_answer(llm_response)
    
    except Exception as e:
        llm_answer = f"API_ERROR: {str(e)[:100]}..."
        excepts.append(idx)
    
    # End timing
    end = time.time() - beg
    
    # Check if the answer is correct
    is_correct = llm_answer == correct_answer
    
    # Add result to results list
    results.append({
        'id': mcq['id'],
        'correct_option': correct_answer,
        'llm_response': llm_answer,
        'is_correct': is_correct,
        'time': end,
        'full_response': llm_response[:100] if len(llm_response) > 100 else llm_response  # Store truncated response for debugging
    })

# Save results to output file
with open(output_file, 'w') as file:
    json.dump(results, file, indent=4)

# Calculate and display overall accuracy
correct_count = sum(1 for r in results if r['is_correct'])
total_count = len(results)
accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

print(f"Processing complete. Results saved to {output_file}")
print(f"Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")

# Save exceptions to separate file
with open('excepts_'+output_file, 'w') as file:
    json.dump(excepts, file, indent=4)

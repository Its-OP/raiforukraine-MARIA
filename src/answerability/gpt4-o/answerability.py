import json
import argparse
import sys
import time
import os
import re
import requests
from openai import OpenAI
from tqdm import tqdm  # For progress bar

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model", type=str, default='gpt-4o')
parser.add_argument('-t', "--temperature", type=float, default=0.5)
parser.add_argument('-p', "--provider", type=str, help="Specify a provider (e.g., 'OpenAI', 'Azure')")
parser.add_argument('-v', "--version", type=str, default="latest", help="Specify GPT model version")
args = parser.parse_args()
input_file = 'mcqs_answerability.json'
output_file = 'answerability'

# Define GPT-4 models
if args.model == 'gpt-4o':
    model = 'gpt-4o'
    output_file += f'_gpt4o_{args.temperature}.json'
elif args.model == 'gpt-4':
    model = 'gpt-4'
    output_file += f'_gpt4_{args.temperature}.json'
elif args.model == 'gpt-4-turbo':
    model = 'gpt-4-turbo'
    output_file += f'_gpt4_turbo_{args.temperature}.json'
elif args.model == 'gpt-4-vision':
    model = 'gpt-4-vision-preview'
    output_file += f'_gpt4_vision_{args.temperature}.json'
else:
    sys.exit('ERROR: Invalid model specified!')

# Update output filename to include provider and version info
if args.provider:
    output_file = output_file.replace('.json', f'_{args.provider}_{args.version}.json')

# Initialize OpenAI API client
if args.provider and args.provider.lower() == 'azure':
    # Azure OpenAI setup
    client = OpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        base_url=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    )
    # For Azure, the deployment name is used instead of the model name
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", model)
else:
    # Standard OpenAI setup
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

# Load MCQs from input file
with open(input_file, 'r') as file:
    mcqs = json.load(file)

#mcqs = mcqs[:10]
    
results = []
excepts = []

# Function to extract JSON from response
def extract_json_answer(response_text):
    """Extract answer from response, handling potential JSON parsing issues"""
    if not response_text:
        return "ERROR"
    
    # Check if the full_response field contains a valid answer but JSON parsing failed
    if isinstance(response_text, str) and "PARSE_ERROR" in response_text and "answer" in response_text:
        patterns = [
            r'"answer"\s*:\s*"([A-D])"',     # "answer": "A"
            r'"answer"\s*:\s*([A-D])',       # "answer": A
            r"'answer'\s*:\s*'([A-D])'",     # 'answer': 'A'
            r'answer\s*:\s*"([A-D])"',       # answer: "A"
            r'answer\s*:\s*([A-D])',         # answer: A
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    
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
for idx, mcq in tqdm(enumerate(mcqs), total=len(mcqs), desc=f"Processing {args.model} questions"):
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
    
    # Initialize llm_response here to ensure it's always defined
    llm_response = ""
    llm_answer = "ERROR"
    
    try:
        # Create request parameters for GPT-4
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions with JSON objects."},
                {"role": "user", "content": prompt_content}
            ],
            "temperature": args.temperature,
            "response_format": {"type": "json_object"}
        }
        
        # Make the API call
        response = client.chat.completions.create(**request_params)
        
        # Extract the response
        llm_response = response.choices[0].message.content
        
        # Ensure llm_response is a string
        if llm_response is None:
            llm_response = ""
            
        # Extract JSON answer
        llm_answer = extract_json_answer(llm_response)
    
    except Exception as e:
        llm_answer = f"API_ERROR: {str(e)[:100]}..."
        excepts.append(idx)
    
    # End timing
    end = time.time() - beg
    
    # Check if the answer is correct
    is_correct = False
    if llm_answer in ['A', 'B', 'C', 'D']:
        is_correct = (llm_answer == correct_answer)
    
    # If we have an API error but full_response contains valid JSON with an answer,
    # try to use that for determining correctness
    if not is_correct and llm_answer.startswith("API_ERROR") and "{" in llm_response and "}" in llm_response:
        try:
            # Extract and use the JSON from full_response
            json_part = llm_response[llm_response.find('{'):llm_response.rfind('}')+1]
            answer_obj = json.loads(json_part)
            if 'answer' in answer_obj:
                extracted_answer = answer_obj['answer'].strip().upper()
                if extracted_answer in ['A', 'B', 'C', 'D']:
                    is_correct = (extracted_answer == correct_answer)
                    # Update llm_answer with the extracted answer
                    llm_answer = extracted_answer
        except:
            pass
    
    # Add result to results list
    results.append({
        'id': mcq['id'],
        'correct_option': correct_answer,
        'llm_response': llm_answer,
        'is_correct': is_correct,
        'time': end,
        'full_response': llm_response
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
    

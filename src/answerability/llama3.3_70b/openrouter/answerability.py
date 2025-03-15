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
parser.add_argument('-l', "--llm", type=str, default='Llama3.3_70b')
parser.add_argument('-t', "--temperature", type=float, default=0.5)
parser.add_argument('-p', "--provider", type=str, help="Specify a provider (e.g., 'Fireworks', 'Anthropic', 'Mistral')")
parser.add_argument('-q', "--quantization", type=str, default="fp16", help="Specify quantization (e.g., 'fp16', 'q8_0')")
args = parser.parse_args()
input_file = 'mcqs_answerability.json'
output_file = 'answerability'

# Define models and provider preferences for OpenRouter API
model_name = None
provider_config = None

if args.llm == 'Llama8':
    model = 'meta-llama/llama-3-8b-instruct'
    output_file += f'_Llama_8_{args.temperature}.json'
    provider_config = {
        "order": [args.provider or "Fireworks"],
        "quantizations": [args.quantization]
    }
elif args.llm == 'Llama3.3_70b':
    model = 'meta-llama/llama-3.3-70b-instruct'
    output_file += f'_Llama_3.3_70b_{args.temperature}.json'
    provider_config = {
        "order": [args.provider or "Fireworks"],
        "quantizations": [args.quantization]
    }
elif args.llm == 'Llama3.1_405b':
    model = 'meta-llama/meta-llama-3.1-405b-instruct'
    output_file += f'_Llama_3.1_405b_{args.temperature}.json'
    provider_config = {
        "order": [args.provider or "Anthropic"],
        "quantizations": [args.quantization]
    }
else:
    sys.exit('ERROR Dataset !!!')

# Update output filename to include provider and quantization info
if args.provider:
    output_file = output_file.replace('.json', f'_{args.provider}_{args.quantization}.json')

# Initialize OpenRouter API client
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://uness.fr",  # Replace with your site
        "X-Title": "MCQ Evaluation Script"
    }
)

# Load MCQs from input file
with open(input_file, 'r') as file:
    mcqs = json.load(file)

#mcqs = mcqs[:10]
    
results = []
excepts = []

# Function to extract JSON from response
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
    
    # Initialize llm_response here to ensure it's always defined
    llm_response = ""
    llm_answer = "ERROR"
    
    try:
        # Make API request to OpenRouter
        # First create the base request parameters
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions with JSON objects."},
                {"role": "user", "content": prompt_content}
            ],
            "temperature": args.temperature,
            "response_format": {"type": "json_object"}
        }
        
        # If we have provider config, use a more direct approach with the underlying library
        if provider_config:
            # Convert parameters to plain dict for direct API call
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                "HTTP-Referer": "https://your-site.com",  # Replace with your site
                "X-Title": "MCQ Evaluation Script"
            }
            
            # Add provider to the request JSON
            request_json = request_params.copy()
            request_json["provider"] = provider_config
            
            # Make direct request to OpenRouter API
            import requests
            openrouter_response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=request_json
            )
            
            if openrouter_response.status_code != 200:
                raise Exception(f"OpenRouter API error: {openrouter_response.text}")
                
            response_data = openrouter_response.json()
            llm_response = response_data["choices"][0]["message"]["content"]
        else:
            # Use the OpenAI client for standard requests without provider parameter
            openai_response = client.chat.completions.create(**request_params)
            llm_response = openai_response.choices[0].message.content
        
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

import json
import argparse
import sys
import time
import os
from openai import OpenAI
from tqdm import tqdm  # For progress bar
from langchain_core.prompts import ChatPromptTemplate

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

results = []
excepts = []

# Template for prompting
template = "Answer the following {subject} multiple-choice question:\n\nQuestion: {question}\nOptions:\n{options}\n\n"
template += "Provide only the letter corresponding to the correct answer (A, B, C, or D) without additional information in this format:'Answer':'<Letter of the answer>'"

# Process each MCQ with progress bar
for idx, mcq in tqdm(enumerate(mcqs), total=len(mcqs), desc=f"Processing {args.llm} questions"):
    question = mcq['question']
    options = f"A: {mcq['opa']}\nB: {mcq['opb']}\nC: {mcq['opc']}\nD: {mcq['opd']}"
    correct_answer = mcq['cop']
    
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
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt_content}
            ],
            temperature=args.temperature
        )
        
        # Extract the response
        llm_response = response.choices[0].message.content
        
        # Process the response to get just the answer letter
        try:
            # Try to parse as JSON if it's in the expected format
            if "{" in llm_response and "}" in llm_response:
                # Extract JSON-like content
                json_part = llm_response[llm_response.find("{"):llm_response.rfind("}")+1]
                llm_response = eval(json_part)
                if 'Answer' in llm_response:
                    llm_answer = llm_response['Answer']
                elif 'answer' in llm_response:
                    llm_answer = llm_response['answer']
                else:
                    llm_answer = llm_response
            else:
                # If it's just the letter, use it directly
                llm_answer = llm_response.strip().upper()
                # Extract just the letter if there's other text
                if len(llm_answer) > 1 and any(letter in llm_answer for letter in ['A', 'B', 'C', 'D']):
                    for letter in ['A', 'B', 'C', 'D']:
                        if letter in llm_answer:
                            llm_answer = letter
                            break
        except:
            llm_answer = llm_response
            excepts.append(idx)
    
    except Exception as e:
        llm_answer = f"ERROR: {str(e)}"
        excepts.append(idx)
    
    # End timing
    end = time.time() - beg
    
    # Check if the answer is correct
    is_correct = llm_answer == correct_answer.upper()
    
    # Add result to results list
    results.append({
        'id': mcq['id'],
        'correct_option': correct_answer,
        'llm_response': llm_answer,
        'is_correct': is_correct,
        'time': end
    })

# Save results to output file
with open(output_file, 'w') as file:
    json.dump(results, file, indent=4)

# Save exceptions to separate file
with open('excepts_'+output_file, 'w') as file:
    json.dump(excepts, file, indent=4)

print(f"Processing complete. Results saved to {output_file}")

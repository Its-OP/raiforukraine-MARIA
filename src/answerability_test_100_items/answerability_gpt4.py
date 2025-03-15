import json
import argparse
import sys
import time
import os
from openai import OpenAI
from tqdm import tqdm  # For progress bar

parser = argparse.ArgumentParser()
parser.add_argument('-l', "--llm", type=str, default='GPT4')
parser.add_argument('-t', "--temperature", type=float, default=0.5)
args = parser.parse_args()
input_file = 'mcqs_answerability.json'
output_file = 'answerability'

# Define models for OpenAI API
model_name = None
if args.llm == 'GPT4':
    output_file += f'_gpt4_{args.temperature}.json'
    model_name = 'gpt-4'
elif args.llm == 'GPT4Turbo':
    output_file += f'_gpt4turbo_{args.temperature}.json'
    model_name = 'gpt-4-turbo'
elif args.llm == 'GPT4o':
    output_file += f'_gpt4o_{args.temperature}.json'
    model_name = 'gpt-4o'
elif args.llm == 'GPT35Turbo':
    output_file += f'_gpt35turbo_{args.temperature}.json'
    model_name = 'gpt-3.5-turbo'
# Keep the existing models for compatibility
elif args.llm == 'Mistral':
    output_file += f'_mistral_16_{args.temperature}.json'
    model_name = 'mistral-small:24b-instruct-2501-fp16'
    # Using Together API
    client = OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
elif args.llm in ['Instruct', 'Llama8', 'Llama16', 'Llama3.3_70b', 'Llama3.1_405b']:
    # Handle other models as before
    if args.llm == 'Instruct':
        model_name = 'mistral-small:24b-instruct-2501-q8_0'
        output_file += f'_mistral_8_{args.temperature}.json'
    elif args.llm == 'Llama8':
        model_name = 'llama3.1:8b-instruct-q8_0'
        output_file += f'_Llama_8_{args.temperature}.json'
    elif args.llm == 'Llama16':
        model_name = 'llama3.1:8b-instruct-fp16'
        output_file += f'_Llama_16_{args.temperature}.json'
    elif args.llm == 'Llama3.3_70b':
        model_name = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
        output_file += f'_Llama_3.3_70b_{args.temperature}.json'
    elif args.llm == 'Llama3.1_405b':
        model_name = 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'
        output_file += f'_Llama_3.1_405b_{args.temperature}.json'
    
    # Using Together API
    client = OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
else:
    sys.exit('ERROR: Unknown model specified!')

# Initialize OpenAI client for GPT models
if args.llm in ['GPT4', 'GPT4Turbo', 'GPT4o', 'GPT35Turbo']:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        # No base_url needed for OpenAI's default API
    )

# Load MCQs from input file
with open(input_file, 'r') as file:
    mcqs = json.load(file)

mcqs = mcqs[:100]
    
results = []
excepts = []

# Template for prompting with structured output
template = "Answer the following {subject} multiple-choice question:\n\nQuestion: {question}\nOptions:\n{options}\n\n"
template += "Provide only the letter corresponding to the correct answer (A, B, C, or D)."

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
        # Make API request with structured output for OpenAI models
        if args.llm in ['GPT4', 'GPT4Turbo', 'GPT4o', 'GPT35Turbo']:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt_content}
                ],
                temperature=args.temperature,
                response_format={"type": "json_object"},
                json_schema={
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"]
                        }
                    },
                    "required": ["answer"]
                }
            )
        else:
            # For non-OpenAI models that don't support response_format
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
            # For OpenAI models with structured output
            if args.llm in ['GPT4', 'GPT4Turbo', 'GPT4o', 'GPT35Turbo']:
                try:
                    # Parse the JSON response
                    import json
                    answer_data = json.loads(llm_response)
                    llm_answer = answer_data.get('answer', '')
                    # Ensure uppercase for consistency in comparison
                    llm_answer = llm_answer.upper()
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    llm_answer = llm_response.strip().upper()
                    if llm_answer in ['A', 'B', 'C', 'D']:
                        pass  # Already correct format
                    else:
                        # Log the issue for debugging
                        print(f"Warning: Failed to parse JSON response: {llm_response}")
                        llm_answer = "ERROR"
                        excepts.append(idx)
            else:
                # For non-OpenAI models, use the previously improved parsing logic
                # Try to parse as JSON if it's in the expected format
                if "{" in llm_response and "}" in llm_response:
                    # Extract JSON-like content
                    json_part = llm_response[llm_response.find("{"):llm_response.rfind("}")+1]
                    try:
                        import json
                        parsed_json = json.loads(json_part)
                        if 'Answer' in parsed_json:
                            llm_answer = parsed_json['Answer']
                        elif 'answer' in parsed_json:
                            llm_answer = parsed_json['answer']
                        else:
                            llm_answer = llm_response
                    except:
                        # Fallback to eval if json.loads fails
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
                    
                    # First try to extract direct answer pattern like "Answer: B" or similar patterns
                    answer_indicators = ["ANSWER:", "ANSWER IS", "THE ANSWER IS", "CORRECT ANSWER IS", "CORRECT ANSWER:"]
                    for indicator in answer_indicators:
                        if indicator in llm_answer:
                            pos = llm_answer.find(indicator) + len(indicator)
                            # Look for A, B, C, or D after the indicator
                            for i in range(pos, min(pos + 20, len(llm_answer))):
                                if llm_answer[i] in ['A', 'B', 'C', 'D']:
                                    llm_answer = llm_answer[i]
                                    break
                            if llm_answer in ['A', 'B', 'C', 'D']:
                                break
                    
                    # If still not a single letter, check if it's just a letter surrounded by text
                    if len(llm_answer) > 1:
                        # If the response is exactly one of the option letters
                        if llm_answer in ['A', 'B', 'C', 'D']:
                            pass  # Already the correct format
                        # Last resort: count occurrences of each letter and take the most frequent one
                        # that appears less than 5 times (to avoid common letters in text)
                        else:
                            counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
                            standalone_letter = None
                            
                            # Look for standalone "A.", "B.", "C.", "D." patterns
                            for letter in ['A', 'B', 'C', 'D']:
                                pattern = letter + "."
                                if pattern in llm_answer and llm_answer.count(pattern) == 1:
                                    standalone_letter = letter
                                    break
                                    
                            # If found a clear standalone letter, use it
                            if standalone_letter:
                                llm_answer = standalone_letter
                            # Otherwise count letters and find the least common one (likely the answer)
                            else:
                                for char in llm_answer:
                                    if char in counts:
                                        counts[char] += 1
                                
                                # Filter for letters that appear at least once but not too frequently
                                answer_candidates = [(letter, count) for letter, count in counts.items() 
                                                   if 1 <= count <= 5]
                                
                                if answer_candidates:
                                    # Sort by frequency (least frequent first) as answers usually appear fewer times
                                    answer_candidates.sort(key=lambda x: x[1])
                                    llm_answer = answer_candidates[0][0]
                
            # Final validation - ensure the answer is one of A, B, C, or D
            if llm_answer not in ['A', 'B', 'C', 'D']:
                print(f"Warning: Invalid answer format: {llm_answer}, from response: {llm_response[:100]}")
                llm_answer = "ERROR"
                excepts.append(idx)
        except:
            llm_answer = llm_response
            excepts.append(idx)
    
    except Exception as e:
        llm_answer = f"ERROR: {str(e)}"
        excepts.append(idx)
    
    # End timing
    end = time.time() - beg
    
    # Check if the answer is correct (only if we have a valid answer)
    is_correct = llm_answer == correct_answer.upper() if llm_answer in ['A', 'B', 'C', 'D'] else False
    
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

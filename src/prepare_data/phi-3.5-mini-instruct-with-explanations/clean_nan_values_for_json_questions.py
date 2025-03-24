import json
import pandas as pd
import numpy as np

def filter_questions_without_nan(input_file, output_file=None):
    """
    Filter out JSON questions with any NaN values
    
    Parameters:
    input_file (str): Path to input JSON file
    output_file (str, optional): Path to save filtered JSON file
    
    Returns:
    list: Filtered list of questions without NaN values
    """
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Flatten the nested JSON structure and identify questions without NaN
    def is_question_complete(item):
        # Check if question dictionary exists
        question = item.get('question', {})
        
        # List of required keys to check for NaN
        required_keys = [
            'question', 
            'option_a', 
            'option_b', 
            'option_c', 
            'option_d', 
            'correct_option'
        ]
        
        # Check if any required key is missing or NaN
        for key in required_keys:
            if not question.get(key):
                return False
        
        return True
    
    # Filter questions
    filtered_data = [item for item in data if is_question_complete(item)]
    
    # Optional: Save filtered data to a new JSON file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(filtered_data, file, ensure_ascii=False, indent=2)
    
    return filtered_data

def main():
    # Replace with your input file path
    input_file = './test_generated_questions.json'
    
    # Optional: Specify output file path
    output_file = './test_generated_questions-filtered.json'
    
    # Filter questions
    filtered_questions = filter_questions_without_nan(input_file, output_file)
    
    # Print some information
    print(f"Total questions in original file: {len(input_file)}")
    print(f"Number of questions after filtering: {len(filtered_questions)}")
    
    # Optional: Print first filtered question for verification
    if filtered_questions:
        print("\nFirst filtered question:")
        print(json.dumps(filtered_questions[0], indent=2))

if __name__ == "__main__":
    main()

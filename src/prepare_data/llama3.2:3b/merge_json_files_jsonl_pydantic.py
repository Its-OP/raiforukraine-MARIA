#!/usr/bin/env python3
"""
Merge JSON Files to JSONL with Pydantic Validation

This script finds all JSON files in a specified directory, validates their structure
using Pydantic models, and merges valid files into a single JSONL file.
"""

import os
import json
import argparse
from typing import List, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError

# Define Pydantic models to validate JSON structure
class MCQQuestion(BaseModel):
    question: str = Field(description="The multiple-choice question")
    option_a: str = Field(description="The first answer option labeled 'A'")
    option_b: str = Field(description="The second answer option labeled 'B'")
    option_c: str = Field(description="The third answer option labeled 'C'")
    option_d: str = Field(description="The fourth answer option labeled 'D'")
    correct_option: str = Field(description="This consists only a letter of the correct option")

class QuestionFile(BaseModel):
    folder: str = Field(description="The folder identifier")
    content: str = Field(description="The content text")
    question: MCQQuestion = Field(description="The MCQ question with options")

def merge_json_files_to_jsonl(input_dir, output_file, log_invalid=False, pretty_print=False):
    """
    Merge all valid JSON files from input_dir into a single JSONL file.
    
    Args:
        input_dir (str): Directory containing individual JSON files
        output_file (str): Path to the output JSONL file
        log_invalid (bool): Whether to write invalid files to a separate log
        pretty_print (bool): Whether to print progress information
    
    Returns:
        tuple: (processed_count, invalid_count)
    """
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0, 0
    
    processed_count = 0
    invalid_count = 0
    invalid_log = []
    
    # Create progress bar if pretty_print is enabled
    files_iter = tqdm(json_files, desc="Validating and merging files") if pretty_print else json_files
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in files_iter:
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    data = json.load(infile)
                    
                    # Validate data structure using Pydantic
                    valid_data = QuestionFile.model_validate(data)
                    
                    # Write as a single line in the output file
                    json.dump(valid_data.model_dump(), outfile)
                    outfile.write('\n')  # Add newline for JSONL format
                    processed_count += 1
                    
            except ValidationError as ve:
                if pretty_print:
                    print(f"Validation Error in {filename}: Invalid structure")
                if log_invalid:
                    invalid_log.append({"file": filename, "error": f"Validation error: {str(ve)}"})
                invalid_count += 1
                
            except json.JSONDecodeError:
                if pretty_print:
                    print(f"Error: {filename} contains invalid JSON. Skipping.")
                if log_invalid:
                    invalid_log.append({"file": filename, "error": "Invalid JSON format"})
                invalid_count += 1
                
            except Exception as e:
                if pretty_print:
                    print(f"Error processing {filename}: {str(e)}")
                if log_invalid:
                    invalid_log.append({"file": filename, "error": str(e)})
                invalid_count += 1
    
    # Write invalid files log if requested
    if log_invalid and invalid_log:
        invalid_log_file = f"{os.path.splitext(output_file)[0]}_invalid.json"
        with open(invalid_log_file, 'w', encoding='utf-8') as logfile:
            json.dump(invalid_log, logfile, indent=2)
        if pretty_print:
            print(f"Wrote log of {len(invalid_log)} invalid files to {invalid_log_file}")
    
    return processed_count, invalid_count

def main():
    parser = argparse.ArgumentParser(description="Merge JSON files into a single JSONL file with validation")
    parser.add_argument("--input-dir", default="generated_questions", 
                        help="Directory containing JSON files (default: generated_questions)")
    parser.add_argument("--output-file", default="merged_questions.jsonl", 
                        help="Output JSONL file (default: merged_questions.jsonl)")
    parser.add_argument("--log-invalid", action="store_true", 
                        help="Log invalid files to a separate JSON file")
    parser.add_argument("--quiet", action="store_true", 
                        help="Suppress progress information")
    
    args = parser.parse_args()
    
    print(f"Merging valid JSON files from {args.input_dir} to {args.output_file}")
    valid_count, invalid_count = merge_json_files_to_jsonl(
        args.input_dir, 
        args.output_file, 
        log_invalid=args.log_invalid,
        pretty_print=not args.quiet
    )
    
    print(f"Successfully merged {valid_count} valid JSON files into {args.output_file}")
    if invalid_count > 0:
        print(f"Skipped {invalid_count} invalid files")

if __name__ == "__main__":
    main()

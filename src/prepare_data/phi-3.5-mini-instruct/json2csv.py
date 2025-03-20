#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to CSV Converter for MCQ Data

Converts JSON formatted multiple-choice questions to CSV format.
"""

import pandas as pd
import argparse
import logging
import json

def parse_content_metadata(content):
    """Extract ID and Rubric from content text."""
    id_value = None
    rubric_value = None
    
    if not isinstance(content, str):
        return id_value, rubric_value
    
    # Extract Identifiant value
    if 'Identifiant=' in content:
        id_parts = content.split('Identifiant=')[1].split('\n')[0].split('|')
        id_value = id_parts[0].strip()
    
    # Extract Rubric value
    if 'Rubric=' in content:
        rubric_parts = content.split('Rubric=')[1].split('\n')[0].split('|')
        rubric_value = rubric_parts[0].strip()
            
    return id_value, rubric_value

def convert_json_to_dataframe(json_path):
    """Convert JSON data to a pandas DataFrame."""
    try:
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        logging.info(f"Loaded JSON with {len(data)} items")
        
        rows = []
        for item in data:
            try:
                # Extract the required fields
                folder = item.get('folder', '')
                content = item.get('content', '')
                question_data = item.get('question', {})
                
                # Skip if missing key data
                if not content or not question_data:
                    continue
                
                # Extract ID and Rubric from content
                id_value, rubric_value = parse_content_metadata(content)
                
                # Create row data
                row = {
                    'id': id_value,
                    'mcq_json': json.dumps(question_data),
                    'rubric': rubric_value,
                }
                
                # Add question and options
                if isinstance(question_data, dict):
                    row['question'] = question_data.get('question', '')
                    row['option_a'] = question_data.get('option_a', '')
                    row['option_b'] = question_data.get('option_b', '')
                    row['option_c'] = question_data.get('option_c', '')
                    row['option_d'] = question_data.get('option_d', '')
                    row['correct_option'] = question_data.get('correct_option', '').lower()
                
                rows.append(row)
            except Exception as e:
                logging.warning(f"Error processing item: {str(e)}")
                continue
        
        logging.info(f"Processed {len(rows)} valid rows")
        
        if not rows:
            logging.error("No valid data rows were extracted")
            return pd.DataFrame()
        
        # Create DataFrame from rows
        cols = ['id', 'mcq_json', 'rubric', 'question', 'option_a', 'option_b', 
                'option_c', 'option_d', 'correct_option']
        result_df = pd.DataFrame(rows)
        
        # Ensure all expected columns exist
        for col in cols:
            if col not in result_df.columns:
                result_df[col] = None
        
        # Return DataFrame with columns in specific order
        return result_df[cols]
    
    except Exception as e:
        logging.error(f"Error converting JSON to DataFrame: {str(e)}")
        raise

def filter_by_ids(df, filter_csv_path):
    """Filter DataFrame based on IDs from another CSV."""
    try:
        filter_df = pd.read_csv(filter_csv_path)
        valid_ids = filter_df.id.unique()
        filtered_df = df[df.id.isin(valid_ids)]
        logging.info(f"Filtered from {len(df)} to {len(filtered_df)} rows")
        return filtered_df
    except Exception as e:
        logging.error(f"Error during filtering: {str(e)}")
        return df

def main():
    """Main function to process command line arguments and convert JSON to CSV."""
    parser = argparse.ArgumentParser(description='Convert MCQ JSON data to CSV format')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--filter', '-f', help='CSV file containing IDs to filter by')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Convert JSON to DataFrame
        logging.info(f"Processing JSON file: {args.input}")
        df = convert_json_to_dataframe(args.input)
        
        if df.empty:
            logging.error("No data extracted from JSON")
            return 1
        
        logging.info(f"Extracted DataFrame with {len(df)} rows")
        
        # Apply ID filtering if specified
        if args.filter:
            logging.info(f"Filtering based on IDs from: {args.filter}")
            df = filter_by_ids(df, args.filter)
        
        # Save to CSV
        logging.info(f"Saving CSV file to: {args.output}")
        df.to_csv(args.output, index=False)
        
        print(f"Successfully converted {len(df)} records to CSV.")
    
    except Exception as e:
        logging.error(f"Error during conversion: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

import concurrent.futures
import time
import os
import json
from tqdm import tqdm

def generate_question_parallel(item, chain):
    try:
        generated_question = chain.invoke({"query": item['content']})
        return {
            "folder": item['folder'],
            "content": item['content'],
            "question": generated_question
        }
    except Exception as e:
        print(f"Error occurred for item in folder {item['folder']}: {e}")
        return None

def save_question_to_file(question_data, output_dir="generated_questions"):
    """Save a single question to its own JSON file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a unique filename based on folder and a timestamp
    folder_name = question_data["folder"]
    timestamp = int(time.time() * 1000)  # millisecond precision
    filename = f"{folder_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(question_data, f, indent=2)
    
    return filepath

def process_batch(batch, chain, batch_size=50, delay_between_batches=60):
    """Process a batch of items with rate limiting"""
    results = []
    
    # Split the batch into smaller sub-batches
    for i in range(0, len(batch), batch_size):
        sub_batch = batch[i:i+batch_size]
        
        # Process the current sub-batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(generate_question_parallel, item, chain): item for item in sub_batch}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    # Save each valid question to its own file
                    filepath = save_question_to_file(result)
                    print(f"Saved question to {filepath}")
                    results.append(result)
        
        # Wait before processing the next batch (unless this is the last batch)
        if i + batch_size < len(batch):
            print(f"Batch completed. Waiting {delay_between_batches} seconds before next batch...")
            time.sleep(delay_between_batches)
    
    return results

def run_in_parallel(dataset, chain, batch_size=50, delay_between_batches=60, max_workers=10):
    """Process the entire dataset in batches with rate limiting"""
    all_results = []
    
    # Calculate the number of items per batch based on max_workers
    items_per_batch = max_workers * 10  # Process 10 items per worker in each major batch
    
    # Create a progress bar for the entire dataset
    with tqdm(total=len(dataset), desc="Generating MCQs") as pbar:
        for i in range(0, len(dataset), items_per_batch):
            # Get the current batch
            batch = dataset[i:i+items_per_batch]
            
            # Process the batch
            batch_results = process_batch(
                batch, 
                chain, 
                batch_size=max_workers,  # Number of concurrent workers
                delay_between_batches=delay_between_batches
            )
            
            all_results.extend(batch_results)
            pbar.update(len(batch))
            
            print(f"{len(all_results)} valid questions generated so far...")
    
    return all_results

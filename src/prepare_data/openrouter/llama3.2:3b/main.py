import json
from mcq_model import create_prompt_chain
from parallel_generation import run_in_parallel
from utils import load_test_set, get_all_txt_contents_from_folders


def main():
    path_lisa = "../../../data/lisa_sheets/"
    path_test_folders = "../../../data/train_test_split/test_folders.json"
    path_train_folders = "../../../data/train_test_split/train_folders.json"

    all_txt_contents = get_all_txt_contents_from_folders(path_lisa)
    
    test_set = load_test_set(all_txt_contents, path_folders=path_test_folders)
    # Optionally uncomment to process train set as well
    # train_set = load_test_set(all_txt_contents, path_folders=path_train_folders)
    
    chain = create_prompt_chain()
    
    # Process with rate limiting - 50 items per batch, 60 seconds between batches
    test_generated_questions = run_in_parallel(
        test_set, 
        chain, 
        batch_size=25,  # Process 50 items at a time
        delay_between_batches=60,  # Wait 60 seconds between batches
        max_workers=50  # Use a maximum of 50 concurrent workers
    )
    
    # Still save the aggregated results for backup/reference
    with open("all_test_generated_questions.json", "w") as f:
        json.dump(test_generated_questions, f, indent=2)
    
    print(f"Generated and saved {len(test_generated_questions)} valid questions.")


if __name__ == "__main__":
    main()

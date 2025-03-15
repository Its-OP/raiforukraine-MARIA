import json
from mcq_model import create_prompt_chain
from parallel_generation import run_in_parallel
from utils import load_test_set, get_all_txt_contents_from_folders


def main():
    path_lisa = "../../data/lisa_sheets/"
    path_test_folders = "../../data/train_test_split/test_folders.json"
    path_train_folders = "../../data/train_test_split/train_folders.json"

    all_txt_contents = get_all_txt_contents_from_folders(path_lisa)
    
    test_set = load_test_set(all_txt_contents, path_folders=path_test_folders)
    train_set = load_test_set(all_txt_contents, path_folders=path_train_folders)
    
    chain = create_prompt_chain()
    
    test_generated_questions = run_in_parallel(test_set, chain, max_workers=10)
    #test_generated_questions = run_in_parallel(test_set[:10], chain, max_workers=10)
    # train_generated_questions = run_in_parallel(train_set, chain, max_workers=10)
    
    with open("test_generated_questions.json", "w") as f:
        json.dump(test_generated_questions, f, indent=2)
    
    print(f"Generated {len(test_generated_questions)} questions.")


if __name__ == "__main__":
    main()

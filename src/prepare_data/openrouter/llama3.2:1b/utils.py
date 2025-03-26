import os
import json


def get_all_txt_contents_as_list(directory_path):
    all_contents = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            
            try:
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    all_contents.append(content) 
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
    
    return all_contents

def get_all_txt_contents_from_folders(parent_directory):
    all_txt_contents = []
    
    # Loop through each folder in the parent directory
    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)
        
        # Check if it is a directory
        if os.path.isdir(folder_path):
            folder_contents = get_all_txt_contents_as_list(folder_path)
            
            for content in folder_contents:
                all_txt_contents.append({
                    "folder": folder_name,
                    "content": content
                })
    
    return all_txt_contents


def load_test_set(all_txt_contents, path_folders):

    try:
        with open(path_folders, "r") as _folders:
            folders = json.load(_folders)

        folder_set = set(folders)

        data_set = [item for item in all_txt_contents if item.get('folder') in folder_set]

        return data_set

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading folders: {e}")
        return [] 


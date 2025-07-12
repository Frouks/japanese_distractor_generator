import os
from pathlib import Path

def rename_files_to_json_extension(base_directory_path_str, dry_run=True):
    """
    Renames files in subdirectories of base_directory_path_str to have a .json extension
    if they don't already have one.

    Args:
        base_directory_path_str (str): The path to the main directory (e.g., 'jawiki_extracted').
        dry_run (bool): If True, only prints what would be renamed. If False, performs renaming.
    """
    base_dir = Path(base_directory_path_str)

    if not base_dir.is_dir():
        print(f"Error: Directory '{base_dir}' not found.")
        return

    print(f"--- {'DRY RUN' if dry_run else 'ACTUAL RENAMING'} ---")
    print(f"Scanning directory: {base_dir}")
    
    files_to_rename_count = 0
    renamed_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate over subdirectories (AA, AB, AC, etc.)
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            # print(f"  Processing subdirectory: {subdir.name}") # Uncomment for more verbose output
            for item in subdir.iterdir():
                # Check if it's a file
                if item.is_file():
                    # Check if it already ends with .json (case-insensitive)
                    if not item.name.lower().endswith(".json"):
                        new_file_name = item.name + ".json"
                        new_file_path = subdir / new_file_name
                        
                        action_prefix = "WOULD RENAME" if dry_run else "RENAMING"

                        # Safety check: if a file with the new name already exists
                        if new_file_path.exists():
                            print(f"    SKIPPED (target '{new_file_name}' already exists): {subdir.name}/{item.name}")
                            skipped_count += 1
                            continue
                        
                        print(f"    {action_prefix}: {subdir.name}/{item.name}  ->  {new_file_name}")
                        files_to_rename_count +=1 # Count even in dry run for summary

                        if not dry_run:
                            try:
                                item.rename(new_file_path)
                                renamed_count += 1
                            except OSError as e:
                                print(f"    ERROR renaming {item.name} in {subdir.name}: {e}")
                                error_count += 1
                    else:
                        # File already has .json extension
                        skipped_count += 1
    
    print(f"\n--- Summary ({'DRY RUN' if dry_run else 'ACTUAL RENAMING'}) ---")
    if dry_run:
        print(f"Files that would be renamed: {files_to_rename_count}")
    else:
        print(f"Files actually renamed: {renamed_count}")
        if error_count > 0:
            print(f"Errors during renaming: {error_count}")
    print(f"Files skipped (already .json or target existed): {skipped_count}")


if __name__ == "__main__":
    wiki_extracted_base_path = "/Volumes/T7/Bachelorthesis/jawiki_data/jawiki_extracted"
    
    print("This script will iterate through subdirectories of the specified path")
    print("and rename files by appending '.json' if they don't already have that extension.")
    print("\n!!! WARNING !!!")
    print("1. Please ensure the 'wiki_extracted_base_path' variable in this script is correct.")
    print(f"   Currently set to: '{wiki_extracted_base_path}'")
    print("2. IT IS STRONGLY RECOMMENDED TO BACK UP YOUR DATA before running this script in non-dry-run mode.")
    print("3. The script will first perform a DRY RUN. Review its output carefully.")

    if not Path(wiki_extracted_base_path).exists():
        print(f"\nERROR: The path '{wiki_extracted_base_path}' does not exist. Please correct it in the script.")
    else:
        print("\nStarting DRY RUN (no files will be changed yet)...")
        rename_files_to_json_extension(wiki_extracted_base_path, dry_run=True)

        print("\nDry run complete. Review the output above to see what changes would be made.")
        
        confirmation = input("Do you want to proceed with the ACTUAL renaming? (Type 'yes' to confirm): ")
        if confirmation.lower() == 'yes':
            print("\nStarting ACTUAL RENAMING...")
            rename_files_to_json_extension(wiki_extracted_base_path, dry_run=False)
            print("\nACTUAL RENAMING finished.")
        else:
            print("\nRenaming cancelled by user. No files were changed.")
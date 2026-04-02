#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanitises, copies, and cleans up specific directory structures.

This script validates directory paths against a specific format, copies them
to a destination, and then cleans up the source.

The copy operation includes all subdirectories but only copies files
that reside in the leaf subdirectories.

The cleanup operation removes the original leaf directories. Finally, it
cleans the yyyymmdd/HHMMSS level directory and its parent yyyymmdd
directory if they become empty.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# --- Constants ---

# An alternative base directory to search for source directories
# if they aren't found in the path provided.
ALT_SOURCE_PREFIX = Path("/var/www/html/meteor/")

# Path to an external script to be run during the final cleanup phase
# if a directory is not completely emptied.
FETCH_SCRIPT = "/var/www/html/bin/fetch.py"

# Regex to *find* the required directory pattern *within* a given path.
# This allows it to match paths like "./20251017/033903" or "/abs/path/20251017/033903".
# It looks for:
# 1. (?P<date>...): A named group 'date' (YYYYMMDD)
#    - (?:20[0-9]{2}): 20xx (years 2000-2099)
#    - (?:0[1-9]|1[0-2]): 01-12 (months)
#    - (?:0[1-9]|[12][0-9]|3[01]): 01-31 (days)
# 2. /: A literal path separator
# 3. (?P<time>...): A named group 'time' (HHMMSS)
#    - [0-2][0-9]: 00-29 (hour)
#    - [0-5][0-9]: 00-59 (minute)
#    - [0-5][0-9]: 00-59 (second)
# 4. (?:...)?$: An optional part for a deeper station/camera path
#    - /: A literal path separator
#    - (?P<station>[a-zA-Z]+): A named group 'station' (one or more letters)
#    - /cam(?P<cam>[1-9][0-9]*): /cam followed by a named group 'cam' (a number not starting with 0)
DIR_PATTERN = re.compile(
    r"(?P<date>(?:20[0-9]{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01]))/"
    r"(?P<time>[0-2][0-9][0-5][0-9][0-5][0-9])"
    r"(?:/(?P<station>[a-zA-Z]+)/cam(?P<cam>[1-9][0-9]*))?"
)


def find_valid_path_component(dir_str: str) -> Optional[str]:
    """
    Searches a directory string for the valid 'yyyymmdd/HHMMSS...' pattern.

    It also performs a basic security check to reject paths containing
    '..' (directory traversal) or '~' (home directory).

    Args:
        dir_str: The raw directory string provided as an argument.

    Returns:
        The matching path component (e.g., "20251017/033903") if found
        and valid, otherwise None.
    """
    # Security check: disallow relative parent lookups or home dir shortcuts.
    if ".." in dir_str or "~" in dir_str:
        return None
    
    # Search for the pattern anywhere in the string.
    match = DIR_PATTERN.search(dir_str)
    if not match:
        return None
    
    # Return the full substring that matched the pattern
    # (e.g., "20251017/033903" or "20251017/033903/station/cam1")
    return match.group(0)


def process_directory(source: Path, destination: Path, dir_arg: str, dry_run: bool, keep: bool):
    """
    Copies a directory structure, cleans leaf directories, and performs final
    cleanup on the source's yyyymmdd/HHMMSS and yyyymmdd level directories.

    Args:
        source: The full, resolved Path object to the source directory.
        destination: The full, resolved Path object to the destination directory.
        dir_arg: The clean, relative path component (e.g., "20251017/033903")
                 Used to calculate the cleanup depth.
        dry_run: If True, print actions without performing them.
        keep: If True, copy files but do not delete the source; instead,
              print what *would* have been deleted.
    """
    print(f"Processing '{source}' -> '{destination}'")

    # --- Determine Cleanup Targets ---
    # We need to find the paths to the yyyymmdd/HHMMSS and yyyymmdd
    # levels *relative to the source* for the final cleanup.
    # We use the clean `dir_arg` to know how many levels deep the source is.
    
    # Count the parts of the clean path (e.g., "2025/10/17/station/cam1" has 5 parts)
    num_parts = len(Path(dir_arg).parts)
    
    # 'time_level_dir' is the yyyymmdd/HHMMSS directory.
    # We start at the full 'source' path and walk up the tree
    # until we are at the HHMMSS level.
    time_level_dir = source
    if num_parts > 2: # If we are deeper than yyyymmdd/HHMMSS
        # Walk up (num_parts - 2) levels.
        # e.g., if path is yyyymmdd/HHMMSS/station/cam1 (4 parts),
        # we walk up (4 - 2) = 2 times to get to HHMMSS.
        for _ in range(num_parts - 2):
            time_level_dir = time_level_dir.parent
    
    # 'date_level_dir' is the parent of the time-level dir (the yyyymmdd dir)
    date_level_dir = time_level_dir.parent


    # --- Dry Run Simulation ---
    # If --dry-run is specified, print all intended actions and exit.
    if dry_run:
        print(f"  [DRY RUN] Would create directory tree for: {destination}")
        print(f"  [DRY RUN] Would copy files from any leaf subdirectories of '{source}'.")
        print("  [DRY RUN] ---")
        print("  [DRY RUN] Would begin primary cleanup phase (deleting processed leaf dirs).")
        print("  [DRY RUN] ---")
        print("  [DRY RUN] Would begin final cleanup phase on target directory:")
        print(f"    [DRY RUN] Would delete any files inside: {time_level_dir}")
        print(f"    [DRY RUN] If '{time_level_dir}' becomes empty, it would be removed.")
        print(f"    [DRY RUN] If '{date_level_dir}' then becomes empty, it would also be removed.")
        # Show the command that would run if the directory is *not* empty
        print(f"    [DRY RUN] If '{time_level_dir}' still contains directories, would run: {sys.executable} {FETCH_SCRIPT} {time_level_dir.resolve()}")
        return # Stop processing this directory

    leaf_dirs_found = []
    copy_successful = False

    # --- 1. Copy Phase (Real and --keep runs) ---
    try:
        # Before walking, ensure the source directory actually exists.
        if not source.is_dir():
             print(f"❌ Error: Source directory '{source}' does not exist. Aborting.", file=sys.stderr)
             return # Stop processing this directory
        
        # os.walk traverses the directory tree top-down.
        for root_str, dirs, files in os.walk(source):
            root = Path(root_str)
            # Find the path of the current directory relative to the source.
            # e.g., if source is /data/20251017/033903
            # and root is /data/20251017/033903/station/cam1
            # relative_path will be "station/cam1"
            relative_path = root.relative_to(source)
            
            # Create the corresponding directory in the destination.
            current_dest_dir = destination / relative_path
            current_dest_dir.mkdir(parents=True, exist_ok=True)
            
            # A "leaf" directory is one that contains no subdirectories.
            if not dirs:
                # We store this leaf dir for the cleanup phase.
                leaf_dirs_found.append(root)
                # Copy all files from this leaf directory to the destination.
                for f in files:
                    shutil.copy2(root / f, current_dest_dir / f)
        
        # Mark copy as successful to allow cleanup to proceed.
        copy_successful = True
    except (IOError, OSError) as e:
        # If any file operation fails, print an error and abort all cleanup.
        print(f"❌ Error during copy from '{source}': {e}. Aborting all cleanup.", file=sys.stderr)
        return # Stop processing this directory

    if not copy_successful:
        return # Should be redundant, but ensures no cleanup if copy failed

    # --- 2. --keep Simulation ---
    # If --keep is specified, we skip the deletion phase and instead
    # print all files and directories that *would* have been deleted.
    if keep:
        print("  --keep specified, listing items that would have been deleted:")
        # Iterate over the leaf directories we found during the copy phase.
        # We sort them by path depth (longest path first) so we
        # process children before parents, though for simulation it's less critical.
        for leaf in sorted(leaf_dirs_found, key=lambda p: len(p.parts), reverse=True):
            try:
                # List files that would be deleted
                for item in leaf.iterdir():
                    if item.is_file(): print(f"    [WOULD DELETE] Source file: {item}")
                # List the leaf directory itself (which would be deleted once empty)
                print(f"    [WOULD DELETE] Leaf directory (once empty): {leaf}")
            except OSError as e:
                # Handle cases where the directory might be unreadable
                print(f"  ⚠️ Could not read {leaf} to list files: {e}", file=sys.stderr)
        
        print(f"  ---")
        print(f"  Final cleanup simulation for target: {time_level_dir}")
        try:
            # Simulate final cleanup on the HHMMSS directory
            for item in time_level_dir.iterdir():
                if item.is_file(): print(f"    [WOULD DELETE] Target-level file: {item}")
            print(f"    [WOULD THEN] Check '{time_level_dir}' and remove if it becomes empty.")
            print(f"    [WOULD THEN ALSO] Check '{date_level_dir}' and remove if that becomes empty.")
            print(f"    [OR ELSE] Run {FETCH_SCRIPT} on '{time_level_dir}' if it's not empty after file deletion.")
        except OSError as e:
            print(f"  ⚠️ Could not read {time_level_dir} to list files: {e}", file=sys.stderr)
        
        return # Stop processing this directory

    # --- 3. Primary Cleanup (Only if not --dry-run and not --keep) ---
    print("  ✅ Copy successful. Beginning primary cleanup...")
    # Sort by depth (longest path first) to ensure we delete child
    # directories before their parents.
    for leaf in sorted(leaf_dirs_found, key=lambda p: len(p.parts), reverse=True):
        try:
            # Delete all files within the leaf directory
            for item in leaf.iterdir():
                if item.is_file():
                    item.unlink()
            
            # Now that the leaf dir is empty, walk *up* the tree,
            # removing directories as long as they are empty.
            current_dir = leaf
            # Stop if we reach the root of the filesystem or the directory
            # *above* our main date-level dir (e.g., the parent of yyyymmdd).
            while current_dir != date_level_dir.parent and current_dir.is_dir() and not any(current_dir.iterdir()):
                print(f"    Removing empty source directory: {current_dir}")
                current_dir.rmdir()
                # Move up to the parent directory for the next loop iteration
                current_dir = current_dir.parent
                
        except OSError as e:
            # If we fail to delete a file or dir, log it and move to the next leaf branch.
            print(f"  ⚠️ Error during primary cleanup of {leaf}: {e}. Halting for this branch.", file=sys.stderr)
            continue

    # --- 4. Final Cleanup ---
    print(f"  Performing final cleanup on target directory: {time_level_dir}")

    # It's possible the primary cleanup already removed the time-level
    # directory (e.g., if "20251017/033903" was the *only* thing processed
    # and it was a leaf). Check if it still exists.
    if not time_level_dir.is_dir():
        print(f"  Skipping final cleanup: target directory '{time_level_dir}' was already removed.")
        return

    try:
        # Delete any *files* remaining at the HHMMSS level.
        # Subdirectories (which weren't leaves) will be left alone.
        for item in time_level_dir.iterdir():
            if item.is_file():
                print(f"    Removing file: {item}")
                item.unlink()

        # Check if the HHMMSS directory is *now* empty (i.e., it contained
        # only files, or it was already empty, or its subdirs were
        # removed in the primary cleanup).
        if not any(time_level_dir.iterdir()):
            print(f"    Target cleanup directory '{time_level_dir}' is empty. Removing it.")
            time_level_dir.rmdir()
            
            # After removing the HHMMSS dir, check its parent (yyyymmdd dir)
            if not any(date_level_dir.iterdir()):
                print(f"    Date-level directory '{date_level_dir}' is now empty. Removing it.")
                date_level_dir.rmdir()
        else:
            # If the HHMMSS directory still contains subdirectories,
            # run the external FETCH_SCRIPT on it.
            # We use sys.executable to ensure we use the same Python interpreter
            # that is running this script.
            cmd = [sys.executable, FETCH_SCRIPT, str(time_level_dir.resolve())]
            print(f"    Target cleanup directory still contains subdirectories. Running: {' '.join(cmd)}")
            # Run the command. check=True will raise an error if the script fails.
            subprocess.run(cmd, check=True, capture_output=True, text=True)

    # --- Error Handling for Final Cleanup ---
    except FileNotFoundError as e:
        # This error is raised if subprocess.run can't find an executable.
        print(f"  ❌ Error: A required file for execution could not be found.", file=sys.stderr)
        # Check *which* file was missing.
        if e.filename == sys.executable:
            print(f"  The Python interpreter specified at '{sys.executable}' appears to be invalid or missing.", file=sys.stderr)
        elif e.filename == FETCH_SCRIPT:
            print(f"  The fetch script itself is missing from the expected location: '{FETCH_SCRIPT}'.", file=sys.stderr)
        else:
            print(f"  Details: {e}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        # This error is raised if the FETCH_SCRIPT returns a non-zero exit code.
        print(f"  ❌ Error: The script '{FETCH_SCRIPT}' failed.", file=sys.stderr)
        print(f"  --- STDOUT ---\n{e.stdout}", file=sys.stderr)
        print(f"  --- STDERR ---\n{e.stderr}", file=sys.stderr)
    except OSError as e:
        # Catch-all for other file-related errors (e.g., permissions).
        print(f"  ❌ Error during final cleanup of '{time_level_dir}': {e}", file=sys.stderr)

def main():
    """
    Main execution function.
    Parses command-line arguments and orchestrates the processing
    of each specified directory.
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Sanitise, copy, and clean up specific meteor data directories.",
        epilog="Example: python3 false.py 20250816/005855 --dest-dir /tmp/my_wrongs"
    )
    # Positional argument: one or more directories to process
    parser.add_argument("directories", metavar="DIR", type=str, nargs='+', help="One or more directories to process.")
    # Optional arguments
    parser.add_argument("--dest-dir", metavar="DIR_PATH", type=str, default="/var/www/html/wrongs/", help="Override the default destination directory (default: /var/www/html/wrongs/).")
    parser.add_argument("--dry-run", action="store_true", help="Print actions that would be taken without executing them.")
    parser.add_argument("--keep", action="store_true", help="Do not delete the source after copying, but list what would be deleted.")
    args = parser.parse_args()

    # --- Validate Destination Directory ---
    dest_base_path = Path(args.dest_dir)
    if not dest_base_path.exists() or not dest_base_path.is_dir():
        print(f"❌ Error: Destination directory '{dest_base_path}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1) # Exit the script with an error code

    if args.dry_run:
        print("--- 💧 DRY RUN MODE ENABLED 💧 ---")
        print("No files will be copied or deleted.")
        print("-" * 35)

    # --- Process Each Directory Argument ---
    for dir_arg_raw in args.directories:
        # Validate the raw argument and extract the clean path component
        # (e.g., "20251017/033903")
        dir_arg_clean = find_valid_path_component(dir_arg_raw)

        if not dir_arg_clean:
            # If the path format is invalid or unsafe, skip it.
            print(f"⚠️ Warning: Discarding invalid or unsafe path: '{dir_arg_raw}'", file=sys.stderr)
            continue

        # --- Resolve Source Path ---
        # The source_dir is the *full path* to the directory to be processed.
        # We first try the path exactly as given (Path() can resolve "./", "////", etc.)
        source_dir = Path(dir_arg_raw)
        if not source_dir.is_dir():
            # If not found, try prepending the alternative prefix.
            source_dir = ALT_SOURCE_PREFIX / dir_arg_raw
        
        # If the source doesn't exist in either location, skip it.
        # We skip this check in dry_run mode to allow simulating
        # actions on paths that might not exist yet.
        if not source_dir.is_dir() and not args.dry_run:
            print(f"⚠️ Warning: Source not found for '{dir_arg_raw}' (tried as-is and in '{ALT_SOURCE_PREFIX}'). Skipping.", file=sys.stderr)
            continue
        
        # --- Determine Destination Path ---
        # The destination path is built by appending the *clean* path
        # component to the base destination directory.
        dest_dir = dest_base_path / dir_arg_clean
        
        # --- Process the Directory ---
        # Pass all resolved paths and options to the main processing function.
        process_directory(source_dir, dest_dir, dir_arg_clean, args.dry_run, args.keep)

    print("\n✅ Script finished.")

# Standard Python boilerplate to run the main() function
# when the script is executed directly.
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script cleans a directory structure by removing empty directories and specific
subdirectories that are missing a key file ('event.txt').

It is designed to operate on a directory structure like:
/basedir/YYYYMMDD/hhmmss/.../camX/

The script performs the following cleanup actions:
1.  For each `YYYYMMDD/hhmmss` directory, it scans all subdirectories.
    If NO 'event.txt' file is found anywhere under `hhmmss`, the entire
    `YYYYMMDD/hhmmss` directory is removed recursively.
2.  If at least one 'event.txt' is found, the script will then only remove
    directories at any level that are completely empty. This is handled by
    walking the directory tree from the bottom up.

The script targets `YYYYMMDD` directories within a specific date range:
-   By default, it processes directories from 30 days ago up to (and including) two days ago.
-   The start date can be customized using the --after option.
-   The end date can be customized using the --before option.
"""

import argparse
import os
import shutil
import re
from datetime import datetime, timedelta

# --- Global dictionary to store summary statistics ---
# This dictionary is updated by the perform_action function.
summary = {
    'dirs_removed': 0,
    'trees_removed': 0,
    'dirs_scanned': 0,
    'errors': 0
}

def parse_date(date_string):
    """
    Parses a date string from one of two formats: YYYYMMDD or YYYY-MM-DD.
    
    Args:
        date_string (str): The date string to parse.

    Returns:
        datetime.date: A date object representing the parsed date.

    Raises:
        ValueError: If the date string does not match the expected formats.
    """
    date_string = date_string.strip()
    if '-' in date_string:
        return datetime.strptime(date_string, '%Y-%m-%d').date()
    else:
        return datetime.strptime(date_string, '%Y%m%d').date()

def perform_action(action, path, dry_run, verbose):
    """
    Executes or simulates a filesystem removal operation (rmdir or rmtree).
    This function handles the --dryrun flag, updates the global summary,
    and prints informative messages.

    Args:
        action (str): The action to perform ('rmdir' for empty dirs, 'rmtree' for non-empty).
        path (str): The full path to the directory to remove.
        dry_run (bool): If True, no changes are made to the filesystem.
        verbose (bool): If True, provides more context (though this function always prints).
    """
    global summary
    prefix = "[DRY RUN] " if dry_run else ""
    
    if action == 'rmdir':
        # This action is for removing directories that are confirmed to be empty.
        print(f"{prefix}Removing empty directory: {path}")
        summary['dirs_removed'] += 1
        if not dry_run:
            try:
                os.rmdir(path)
            except OSError as e:
                print(f"  -> ERROR: Could not remove directory {path}: {e}")
                summary['dirs_removed'] -= 1  # Decrement count on failure
                summary['errors'] += 1
    
    elif action == 'rmtree':
        # This action is for recursively removing a directory and all its contents.
        print(f"{prefix}Recursively removing directory tree: {path}")
        summary['trees_removed'] += 1
        if not dry_run:
            try:
                shutil.rmtree(path)
            except OSError as e:
                print(f"  -> ERROR: Could not remove directory tree {path}: {e}")
                summary['trees_removed'] -= 1  # Decrement count on failure
                summary['errors'] += 1

def clean_directory(base_dir, start_date, end_date, dry_run, verbose):
    """
    Main function to scan and clean the specified base directory.

    It iterates through YYYYMMDD directories, then hhmmss directories, applying
    the cleanup logic based on the presence of 'event.txt'.

    Args:
        base_dir (str): The path to the root directory to clean.
        start_date (datetime.date): The earliest date for directories to process.
        end_date (datetime.date): The latest date for directories to process.
        dry_run (bool): If True, simulate actions without making changes.
        verbose (bool): If True, print additional progress information.
    """
    global summary
    date_dir_pattern = re.compile(r'^\d{8}$')
    time_dir_pattern = re.compile(r'^\d{6}$')

    if verbose:
        print(f"Scanning base directory: {base_dir}")
        print(f"Processing directories from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found or is not a directory.")
        return

    # Iterate over all YYYYMMDD directories.
    for yyyymmdd_name in sorted(os.listdir(base_dir)):
        if not (date_dir_pattern.match(yyyymmdd_name) and os.path.isdir(os.path.join(base_dir, yyyymmdd_name))):
            continue
        
        try:
            dir_date = datetime.strptime(yyyymmdd_name, '%Y%m%d').date()
            if not (start_date <= dir_date <= end_date):
                continue
        except ValueError:
            continue

        yyyymmdd_path = os.path.join(base_dir, yyyymmdd_name)
        summary['dirs_scanned'] += 1
        if verbose:
            print(f"\n--- Processing {yyyymmdd_path} ---")
        
        # Iterate over all hhmmss directories inside the YYYYMMDD directory.
        for hhmmss_name in sorted(os.listdir(yyyymmdd_path)):
            hhmmss_path = os.path.join(yyyymmdd_path, hhmmss_name)
            if not (time_dir_pattern.match(hhmmss_name) and os.path.isdir(hhmmss_path)):
                continue

            # Step 1: Check for the existence of any 'event.txt' within this hhmmss tree.
            event_found = False
            for _, _, filenames in os.walk(hhmmss_path):
                if 'event.txt' in filenames:
                    event_found = True
                    if verbose:
                        print(f"  -> Found event.txt in {hhmmss_path}, will keep it.")
                    break  # Found one, no need to search further.

            # Step 2: Act based on whether an event file was found.
            if not event_found:
                # Safeguard: Before removing the tree, do a final check to ensure no event.txt exists.
                # This should ideally never trigger, but it prevents accidental data loss if logic is flawed.
                if verbose:
                    print(f"  -> Performing final safeguard check on {hhmmss_path} before removal.")
                final_check_found = False
                for _, _, filenames in os.walk(hhmmss_path):
                    if 'event.txt' in filenames:
                        final_check_found = True
                        break
                
                if final_check_found:
                    # This block runs for both normal and dryrun modes.
                    print(f"  -> INTERNAL ERROR: Safeguard triggered. 'event.txt' was found in {hhmmss_path} during final check. Directory will NOT be removed.")
                    summary['errors'] += 1
                else:
                    # If no event.txt was found anywhere, remove the entire hhmmss directory.
                    if verbose:
                        print(f"  -> No event.txt found in {hhmmss_path}.")
                    perform_action('rmtree', hhmmss_path, dry_run, verbose)
            else:
                # If an event.txt was found, only clean up genuinely empty directories.
                if verbose:
                    print(f"  -> Cleaning up empty directories inside {hhmmss_path}.")
                for dirpath, _, _ in os.walk(hhmmss_path, topdown=False):
                    try:
                        if not os.listdir(dirpath):
                            perform_action('rmdir', dirpath, dry_run, verbose)
                    except FileNotFoundError:
                        # Can happen if a parent directory was already removed.
                        continue
        
        # After cleaning hhmmss directories, the YYYYMMDD directory might be empty.
        try:
            if not os.listdir(yyyymmdd_path):
                perform_action('rmdir', yyyymmdd_path, dry_run, verbose)
        except FileNotFoundError:
            continue

def main():
    """
    Parses command-line arguments, calculates date ranges, and initiates the
    cleaning process. Prints a final summary of actions taken.
    """
    parser = argparse.ArgumentParser(
        description="Clean up event directories by removing empty folders and specific subdirectories lacking an 'event.txt' file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--basedir',
        default='/home/httpd/norskmeteornettverk.no/meteor',
        help='The base directory to scan for YYYYMMDD subdirectories.\n(default: /home/httpd/norskmeteornettverk.no/meteor)'
    )
    parser.add_argument(
        '--after',
        help='Process directories with dates after this date.\nAccepts YYYYMMDD or YYYY-MM-DD format.\n(default: 30 days ago)'
    )
    parser.add_argument(
        '--before',
        help='Process directories with dates before this date.\nAccepts YYYYMMDD or YYYY-MM-DD format.\n(default: two days ago)'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help="Perform a dry run. The script will only print what it would do\nwithout actually deleting any files or directories."
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose output to see more detailed information about the script's progress."
    )

    args = parser.parse_args()

    # --- Date Calculation ---
    today = datetime.now().date()
    
    # Set the start date based on the --after argument or default to 30 days ago.
    if args.after:
        try:
            start_date = parse_date(args.after)
        except ValueError:
            print(f"Error: Invalid date format for --after. Please use YYYYMMDD or YYYY-MM-DD.")
            exit(1)
    else:
        # Default start date is 30 days before today.
        start_date = today - timedelta(days=30)

    # Set the end date based on the --before argument or default to two days ago.
    if args.before:
        try:
            end_date = parse_date(args.before)
        except ValueError:
            print(f"Error: Invalid date format for --before. Please use YYYYMMDD or YYYY-MM-DD.")
            exit(1)
    else:
        # Default end date is two days ago (i.e., does not include yesterday).
        end_date = today - timedelta(days=2)
    
    # Validate that the start date is not after the end date.
    if start_date > end_date:
        print(f"Error: The --after date ({start_date.strftime('%Y-%m-%d')}) cannot be later than the --before date ({end_date.strftime('%Y-%m-%d')}).")
        exit(1)

    if args.dryrun:
        print("--- DRY RUN MODE ENABLED: No changes will be made to the filesystem. ---")

    # Start the main cleaning process.
    clean_directory(args.basedir, start_date, end_date, args.dryrun, args.verbose)
    
    # --- Final Summary ---
    print("\n" + "="*20)
    print("      SUMMARY")
    print("="*20)
    if args.dryrun:
        print("Operation was a DRY RUN. The summary reflects actions that WOULD have been taken.")
    
    print(f"Date range processed: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"YYYYMMDD directories scanned in range: {summary['dirs_scanned']}")
    print(f"Empty directories removed: {summary['dirs_removed']}")
    print(f"Directory trees removed (e.g., hhmmss without event.txt): {summary['trees_removed']}")
    if summary['errors'] > 0:
        print(f"Errors encountered: {summary['errors']}")
    print("-" * 20)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()


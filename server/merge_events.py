#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, date
import logging

# --- Configuration for Logging ---
# This sets up how messages will be displayed to the user.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Report Class ---
class Report:
    """A simple class to collect data for the final summary report."""
    def __init__(self):
        self.successful_groups = []
        self.failed_groups = []
        self.skipped_groups = []

    def add_success(self, final_path, source_paths):
        self.successful_groups.append({
            "final_path": final_path,
            "sources": source_paths
        })

    def add_failure(self, source_paths, reason):
        self.failed_groups.append({
            "sources": source_paths,
            "reason": reason
        })

    def add_skipped(self, source_paths, reason):
        self.skipped_groups.append({
            "sources": source_paths,
            "reason": reason
        })

    def print_summary(self):
        """Prints the final summary report to the console."""
        print("\n" + "="*80)
        print(" SCRIPT EXECUTION SUMMARY".center(80))
        print("="*80)

        if not self.successful_groups and not self.failed_groups and not self.skipped_groups:
            print("\nNo groups were processed.\n")
            return

        # --- Successful Groups ---
        if self.successful_groups:
            print(f"\n--- Successfully Merged Groups ({len(self.successful_groups)}) ---\n")
            for group in self.successful_groups:
                source_names = [os.path.basename(p) for p in group['sources']]
                dest_name = os.path.basename(group['final_path'])
                date_dir = os.path.basename(os.path.dirname(group['final_path']))
                print(f"  {date_dir}/{dest_name} <= {', '.join(source_names)}")
        
        # --- Skipped Groups ---
        if self.skipped_groups:
            print(f"\n--- Skipped Groups ({len(self.skipped_groups)}) ---\n")
            for group in self.skipped_groups:
                source_names = [os.path.basename(p) for p in group['sources']]
                date_dir = os.path.basename(os.path.dirname(group['sources'][0]))
                print(f"  {date_dir}/{', '.join(source_names)} - Reason: {group['reason']}")

        # --- Failed Groups ---
        if self.failed_groups:
            print(f"\n--- Failed or Incomplete Groups ({len(self.failed_groups)}) ---\n")
            for group in self.failed_groups:
                source_names = [os.path.basename(p) for p in group['sources']]
                date_dir = os.path.basename(os.path.dirname(group['sources'][0]))
                print(f"  Group: {date_dir}/{', '.join(source_names)}")
                print(f"  Reason: {group['reason']}\n")
        
        print("="*80)
        print(" END OF SUMMARY".center(80))
        print("="*80)


def parse_arguments():
    """
    Parses command-line arguments provided by the user.
    This function sets up the expected arguments, their default values, and help messages.
    """
    parser = argparse.ArgumentParser(
        description="Group timestamped directories (YYYYMMDD/HHMMSS) that are close together in time."
    )
    parser.add_argument(
        "directory",
        nargs='?',
        default=os.getcwd(),
        help="The root directory to search for timestamped subdirectories. Defaults to the current directory."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=8,
        help="The maximum time difference in seconds to consider timestamps as part of the same group. Defaults to 8."
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only consider directories with a date later than this. Format: YYYYMMDD or YYYY-MM-DD."
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Only consider directories with a date up to and including this one. Format: YYYYMMDD or YYYY-MM-DD."
    )
    parser.add_argument(
        "--grace",
        type=int,
        default=3600,
        help="Grace period in seconds. If any file in a group is newer than this, the group is skipped. Defaults to 3600."
    )
    parser.add_argument(
        "--exec",
        dest='exec_command',
        type=str,
        default=None,
        help="Path to an executable to run after a group is successfully processed. The final directory path is passed as an argument."
    )
    # The --dry-run option is useful for testing without making actual changes.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a trial run without creating directories or moving files. Just show what would be done."
    )
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.interval > 300:
        logging.error(f"Interval of {args.interval}s is too high. Please choose a value of 300 or less.")
        exit(1)
    if args.interval < 2:
        logging.error(f"Interval of {args.interval}s is too low. Please choose a value of 2 or more.")
        exit(1)
        
    if args.since:
        try:
            since_date = datetime.strptime(args.since, "%Y-%m-%d").date()
        except ValueError:
            try:
                since_date = datetime.strptime(args.since, "%Y%m%d").date()
            except ValueError:
                logging.error(f"Invalid date format for --since: '{args.since}'. Use YYYYMMDD or YYYY-MM-DD.")
                exit(1)
        args.since = since_date

    if args.until:
        try:
            until_date = datetime.strptime(args.until, "%Y-%m-%d").date()
        except ValueError:
            try:
                until_date = datetime.strptime(args.until, "%Y%m%d").date()
            except ValueError:
                logging.error(f"Invalid date format for --until: '{args.until}'. Use YYYYMMDD or YYYY-MM-DD.")
                exit(1)
        args.until = until_date

    if args.grace < 0:
        logging.error(f"Grace period must be a positive number, not {args.grace}.")
        exit(1)

    # Validate the --exec argument before doing any work
    if args.exec_command:
        logging.info(f"Validating executable: {args.exec_command}")
        if not os.path.isfile(args.exec_command):
            logging.error(f"Executable '{args.exec_command}' does not exist or is not a file.")
            exit(1)
        if not os.access(args.exec_command, os.X_OK):
            logging.error(f"File '{args.exec_command}' is not executable.")
            exit(1)
        logging.info("Executable validation passed.")

    return args

def find_timestamp_dirs(root_dir, since_date=None, until_date=None):
    """
    Scans the given directory for subdirectories matching the YYYYMMDD/HHMMSS format.
    """
    date_dir_pattern = re.compile(r'^\d{8}$')
    time_dir_pattern = re.compile(r'^\d{6}$')
    found_dirs = []

    logging.info(f"Scanning for timestamp directories in '{root_dir}'...")
    if since_date:
        logging.info(f"Filtering directories with a date after {since_date.strftime('%Y-%m-%d')}.")
    if until_date:
        logging.info(f"Filtering directories with a date up to {until_date.strftime('%Y-%m-%d')}.")

    if not os.path.isdir(root_dir):
        logging.error(f"Error: The specified directory '{root_dir}' does not exist.")
        return []

    for dirpath, dirnames, _ in os.walk(root_dir):
        dir_base_name = os.path.basename(dirpath)
        if date_dir_pattern.match(dir_base_name):
            date_str = dir_base_name
            for dirname in dirnames:
                if '_' in dirname:
                    continue
                if time_dir_pattern.match(dirname):
                    time_str = dirname
                    try:
                        dt_obj = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                        
                        if since_date and dt_obj.date() <= since_date:
                            continue
                        if until_date and dt_obj.date() > until_date:
                            continue

                        full_path = os.path.join(dirpath, dirname)
                        found_dirs.append((dt_obj, full_path))
                    except ValueError:
                        logging.warning(f"Skipping invalid timestamp format: {date_str}/{time_str}")

    found_dirs.sort()
    logging.info(f"Found {len(found_dirs)} valid timestamp directories matching the criteria.")
    return found_dirs

def group_timestamps(timestamp_dirs, interval_seconds):
    """
    Groups the found timestamp directories based on the time interval.
    """
    if not timestamp_dirs:
        return []

    groups = []
    current_group = [timestamp_dirs[0]]
    time_delta = timedelta(seconds=interval_seconds)

    for i in range(1, len(timestamp_dirs)):
        prev_dt, _ = current_group[-1]
        current_dt, _ = timestamp_dirs[i]
        if current_dt - prev_dt <= time_delta:
            current_group.append(timestamp_dirs[i])
        else:
            groups.append(current_group)
            current_group = [timestamp_dirs[i]]
    
    groups.append(current_group)
    logging.info(f"Formed {len(groups)} groups from the found directories.")
    return groups

def is_group_within_grace_period(group, grace_seconds):
    """
    Checks if all files and directories within a group are older than the grace period.
    This check uses lstat to get the modification time of the symlink itself, not its target.
    """
    now = datetime.now()
    grace_time_limit = now - timedelta(seconds=grace_seconds)

    for _, dir_path in group:
        for root, dirs, files in os.walk(dir_path):
            # Check modification time of all items (files and directories)
            for name in files + dirs:
                path = os.path.join(root, name)
                try:
                    # Use lstat to get info of the file/symlink itself, not the target
                    mtime_epoch = os.lstat(path).st_mtime
                    mtime = datetime.fromtimestamp(mtime_epoch)
                    if mtime > grace_time_limit:
                        logging.warning(f"  - GRACE PERIOD CHECK FAILED: Item '{path}' is too new (modified at {mtime.strftime('%Y-%m-%d %H:%M:%S')}).")
                        return False
                except OSError as e:
                    logging.error(f"  - Could not get modification time for '{path}': {e}")
                    return False # Fail safely
    return True

def merge_directories(src_dir, dst_dir, dry_run=False):
    """
    Recursively merges the contents of src_dir into dst_dir.
    Returns True on success, False on failure.
    """
    logging.info(f"    - Merging contents of '{src_dir}' into '{dst_dir}'")
    
    if dry_run:
        if not os.path.isdir(src_dir):
            logging.warning(f"      - [DRY RUN] Source '{src_dir}' is not a directory. Cannot merge.")
            return True
        for item_name in os.listdir(src_dir):
            logging.info(f"      - [DRY RUN] Would move '{item_name}' to destination.")
        return True

    if not os.path.isdir(dst_dir):
        logging.error(f"      - Merge destination '{dst_dir}' is not a directory. Skipping merge for '{src_dir}'.")
        return False

    for item_name in os.listdir(src_dir):
        src_item_path = os.path.join(src_dir, item_name)
        dst_item_path = os.path.join(dst_dir, item_name)

        if os.path.exists(dst_item_path):
            if os.path.isdir(src_item_path) and os.path.isdir(dst_item_path):
                if merge_directories(src_item_path, dst_item_path, dry_run):
                    try:
                        os.rmdir(src_item_path)
                        logging.info(f"        - Cleaned up merged subdirectory: {src_item_path}")
                    except OSError as e:
                        logging.error(f"        - FAILED to remove merged subdirectory '{src_item_path}': {e}")
                        return False
                else:
                    return False
            elif os.path.isfile(src_item_path) and os.path.isfile(dst_item_path):
                source_mtime = os.path.getmtime(src_item_path)
                dest_mtime = os.path.getmtime(dst_item_path)
                if source_mtime > dest_mtime:
                    logging.warning(f"      - CONFLICT: File '{item_name}' is newer. Replacing.")
                    try:
                        shutil.move(src_item_path, dst_item_path)
                    except (OSError, shutil.Error) as e:
                        logging.error(f"        - FAILED to replace file: {e}")
                        return False
                else:
                    logging.info(f"      - Destination file '{item_name}' is newer or same age. Removing older source file.")
                    try:
                        os.remove(src_item_path)
                    except OSError as e:
                        logging.error(f"        - FAILED to remove older source file '{src_item_path}': {e}")
                        return False
            else:
                logging.error(f"      - TYPE CONFLICT: Cannot merge '{src_item_path}' with '{dst_item_path}'. Skipping.")
        else:
            logging.info(f"      - Moving '{item_name}' to destination.")
            try:
                shutil.move(src_item_path, dst_item_path)
            except (OSError, shutil.Error) as e:
                logging.error(f"        - FAILED to move item: {e}")
                return False
    
    return True

def process_group(group, args, report):
    """
    Processes a single group of timestamp directories. Merges them into a temporary
    directory, removes the sources, and then renames the temp dir to the oldest timestamp.
    """
    if len(group) <= 1:
        return

    source_paths = [path for _, path in group]
    dry_run = args.dry_run

    # --- Grace Period Check ---
    logging.info(f"Performing grace period check for group starting with {os.path.basename(group[0][1])}...")
    if not is_group_within_grace_period(group, args.grace):
        reason = "Group contains recently modified files (within grace period)."
        logging.warning(f"{reason} Group will be skipped.")
        report.add_skipped(source_paths, reason)
        return
    logging.info("  - Grace period check passed.")

    # --- Pre-cleanup "all-or-nothing" permission check (BEFORE any action is taken) ---
    logging.info("Performing pre-cleanup permission checks...")
    if not dry_run:
        for path in source_paths:
            parent_dir = os.path.dirname(path)
            if not os.access(parent_dir, os.W_OK):
                reason = f"No write permission in parent directory '{parent_dir}' to remove '{path}'."
                logging.error(f"  - PRE-CHECK FAILED: {reason} Aborting group processing.")
                report.add_failure(source_paths, reason)
                return
    logging.info("  - Pre-cleanup checks passed.")

    # --- Create Temporary Directory ---
    time_stamps = [os.path.basename(path) for _, path in group]
    temp_dir_name = "_".join(time_stamps)
    _, latest_path = group[-1]
    parent_dir_of_latest = os.path.dirname(latest_path)
    temp_destination_dir = os.path.join(parent_dir_of_latest, temp_dir_name)

    logging.info(f"Processing group -> Merging into temporary directory '{temp_destination_dir}'")

    if dry_run:
        logging.info("[DRY RUN] Would create temporary directory: " + temp_destination_dir)
    else:
        try:
            if os.path.exists(temp_destination_dir):
                logging.warning(f"Removing leftover temporary directory from a previous run: {temp_destination_dir}")
                shutil.rmtree(temp_destination_dir)
            os.makedirs(temp_destination_dir)
        except OSError as e:
            reason = f"Could not create temporary directory '{temp_destination_dir}': {e}"
            logging.error(reason)
            report.add_failure(source_paths, reason)
            return

    # --- Merge all source directories ---
    all_merges_succeeded = True
    for _, source_dir_path in group:
        if not merge_directories(source_dir_path, temp_destination_dir, dry_run):
            all_merges_succeeded = False
            break
    
    if not all_merges_succeeded:
        reason = f"A failure occurred during the merge phase. Merged content remains in '{temp_destination_dir}'."
        logging.error(reason)
        report.add_failure(source_paths, reason)
        return

    # --- Final Cleanup and Rename ---
    all_sources_removed_successfully = True
    if not dry_run:
        for source_dir_path in source_paths:
            try:
                os.rmdir(source_dir_path)
                logging.info(f"  - Cleaned up empty source directory: {source_dir_path}")
            except OSError as e:
                logging.error(f"  - Failed to remove source directory '{source_dir_path}': {e}.")
                all_sources_removed_successfully = False
    else:
        for source_dir_path in source_paths:
            logging.info(f"[DRY RUN] Would remove empty source directory: {source_dir_path}")

    if all_sources_removed_successfully:
        oldest_source_path = source_paths[0]
        logging.info(f"All source directories removed. Renaming '{temp_destination_dir}' to '{oldest_source_path}'.")
        if not dry_run:
            try:
                shutil.move(temp_destination_dir, oldest_source_path)
                report.add_success(oldest_source_path, source_paths)
                
                if args.exec_command:
                    logging.info(f"Executing command: {args.exec_command} {oldest_source_path}")
                    try:
                        subprocess.run([args.exec_command, oldest_source_path], check=True)
                    except FileNotFoundError:
                        logging.error(f"  - EXECUTION FAILED: Command '{args.exec_command}' not found.")
                    except subprocess.CalledProcessError as e:
                        logging.error(f"  - EXECUTION FAILED: Command returned non-zero exit status {e.returncode}.")
                    except Exception as e:
                            logging.error(f"  - EXECUTION FAILED: An unexpected error occurred: {e}")

            except (OSError, shutil.Error) as e:
                reason = f"FAILED to rename final directory: {e}"
                logging.error(f"  - {reason}")
                report.add_failure(source_paths, reason)
        else: # This is a dry run
            logging.info(f"[DRY RUN] Would rename '{temp_destination_dir}' to '{oldest_source_path}'.")
            report.add_success(oldest_source_path, source_paths)
            if args.exec_command:
                logging.info(f"[DRY RUN] Would execute: {args.exec_command} {oldest_source_path}")
    else:
        reason = f"Could not remove all source directories during cleanup. Merged directory remains at: '{temp_destination_dir}'"
        logging.warning(reason)
        report.add_failure(source_paths, reason)


def main():
    """
    The main function to orchestrate the script's operations.
    """
    args = parse_arguments()
    report = Report()
    
    if args.dry_run:
        logging.info("--- Starting in DRY RUN mode. No changes will be made. ---")

    timestamp_dirs = find_timestamp_dirs(args.directory, args.since, args.until)
    
    if not timestamp_dirs:
        logging.info("No timestamp directories found to process. Exiting.")
        report.print_summary()
        return

    groups = group_timestamps(timestamp_dirs, args.interval)
    
    groups_to_process = [g for g in groups if len(g) > 1]
    
    if not groups_to_process:
        logging.info("No groups with multiple timestamps were formed. Nothing to do.")
        report.print_summary()
        return
        
    logging.info(f"Found {len(groups_to_process)} groups to merge.")

    for group in groups_to_process:
        process_group(group, args, report)
        
    logging.info("--- Script finished. ---")
    report.print_summary()


if __name__ == "__main__":
    main()

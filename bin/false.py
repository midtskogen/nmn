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

# --- Constants ---
ALT_SOURCE_PREFIX = Path("/var/www/html/meteor/")
FETCH_SCRIPT = "/var/www/html/bin/fetch.py"

# Regex to validate the two allowed directory formats.
DIR_PATTERN = re.compile(
    r"^(?P<date>(?:20[0-9]{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01]))/"
    r"(?P<time>[0-2][0-9][0-5][0-9][0-5][0-9])"
    r"(?:/(?P<station>[a-zA-Z]+)/cam(?P<cam>[1-9][0-9]*))?$"
)


def sanitize_directory(dir_str: str) -> bool:
    """Validates a directory string against the required pattern."""
    if ".." in dir_str or "~" in dir_str:
        return False
    return bool(DIR_PATTERN.fullmatch(dir_str))


def process_directory(source: Path, destination: Path, dir_arg: str, dry_run: bool, keep: bool):
    """
    Copies a directory structure, cleans leaf directories, and performs final
    cleanup on the source's yyyymmdd/HHMMSS and yyyymmdd level directories.
    """
    print(f"Processing '{source}' -> '{destination}'")

    # Determine the target directories for the final cleanup steps
    num_parts = len(Path(dir_arg).parts)
    time_level_dir = source
    if num_parts > 2:
        for _ in range(num_parts - 2):
            time_level_dir = time_level_dir.parent
    date_level_dir = time_level_dir.parent


    # --- Dry Run Simulation ---
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
        print(f"    [DRY RUN] If '{time_level_dir}' still contains directories, would run: {sys.executable} {FETCH_SCRIPT} {time_level_dir.resolve()}")
        return

    leaf_dirs_found = []
    copy_successful = False

    # --- 1. Copy Phase (Real and --keep runs) ---
    try:
        # Check if source exists before walking
        if not source.is_dir():
             print(f"❌ Error: Source directory '{source}' does not exist. Aborting.", file=sys.stderr)
             return
        for root_str, dirs, files in os.walk(source):
            root = Path(root_str)
            relative_path = root.relative_to(source)
            current_dest_dir = destination / relative_path
            current_dest_dir.mkdir(parents=True, exist_ok=True)
            if not dirs:
                leaf_dirs_found.append(root)
                for f in files:
                    shutil.copy2(root / f, current_dest_dir / f)
        copy_successful = True
    except (IOError, OSError) as e:
        print(f"❌ Error during copy from '{source}': {e}. Aborting all cleanup.", file=sys.stderr)
        return

    if not copy_successful: return

    # --- 2. --keep Simulation ---
    if keep:
        print("  --keep specified, listing items that would have been deleted:")
        for leaf in sorted(leaf_dirs_found, key=lambda p: len(p.parts), reverse=True):
            try:
                for item in leaf.iterdir():
                    if item.is_file(): print(f"    [WOULD DELETE] Source file: {item}")
                print(f"    [WOULD DELETE] Leaf directory (once empty): {leaf}")
            except OSError as e: print(f"  ⚠️ Could not read {leaf} to list files: {e}", file=sys.stderr)
        print(f"  ---")
        print(f"  Final cleanup simulation for target: {time_level_dir}")
        try:
            for item in time_level_dir.iterdir():
                if item.is_file(): print(f"    [WOULD DELETE] Target-level file: {item}")
            print(f"    [WOULD THEN] Check '{time_level_dir}' and remove if it becomes empty.")
            print(f"    [WOULD THEN ALSO] Check '{date_level_dir}' and remove if that becomes empty.")
            print(f"    [OR ELSE] Run {FETCH_SCRIPT} on '{time_level_dir}' if it's not empty after file deletion.")
        except OSError as e: print(f"  ⚠️ Could not read {time_level_dir} to list files: {e}", file=sys.stderr)
        return

    # --- 3. Primary Cleanup ---
    print("  ✅ Copy successful. Beginning primary cleanup...")
    for leaf in sorted(leaf_dirs_found, key=lambda p: len(p.parts), reverse=True):
        try:
            for item in leaf.iterdir():
                if item.is_file(): item.unlink()
            current_dir = leaf
            # Walk up the tree from the leaf, removing empty directories
            while current_dir != date_level_dir.parent and current_dir.is_dir() and not any(current_dir.iterdir()):
                print(f"    Removing empty source directory: {current_dir}")
                current_dir.rmdir()
                current_dir = current_dir.parent
        except OSError as e:
            print(f"  ⚠️ Error during primary cleanup of {leaf}: {e}. Halting for this branch.", file=sys.stderr)
            continue

    # --- 4. Final Cleanup ---
    print(f"  Performing final cleanup on target directory: {time_level_dir}")

    # *** FIX 1: Check if the target directory was already removed in the primary cleanup phase. ***
    if not time_level_dir.is_dir():
        print(f"  Skipping final cleanup: target directory '{time_level_dir}' was already removed.")
        return

    try:
        for item in time_level_dir.iterdir():
            if item.is_file():
                print(f"    Removing file: {item}")
                item.unlink()

        if not any(time_level_dir.iterdir()):
            print(f"    Target cleanup directory '{time_level_dir}' is empty. Removing it.")
            time_level_dir.rmdir()
            # The final requested step: check the parent directory
            if not any(date_level_dir.iterdir()):
                print(f"    Date-level directory '{date_level_dir}' is now empty. Removing it.")
                date_level_dir.rmdir()
        else:
            # *** FIX 2A: Use sys.executable for a robust, explicit call to the interpreter. ***
            cmd = [sys.executable, FETCH_SCRIPT, str(time_level_dir.resolve())]
            print(f"    Target cleanup directory still contains subdirectories. Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)

    # *** FIX 2B: Provide a more detailed and accurate error message for FileNotFoundError. ***
    except FileNotFoundError as e:
        print(f"  ❌ Error: A required file for execution could not be found.", file=sys.stderr)
        # The exception object tells us exactly which file was missing.
        if e.filename == sys.executable:
            print(f"  The Python interpreter specified at '{sys.executable}' appears to be invalid or missing.", file=sys.stderr)
        elif e.filename == FETCH_SCRIPT:
            print(f"  The fetch script itself is missing from the expected location: '{FETCH_SCRIPT}'.", file=sys.stderr)
        else:
            # Fallback for any other unexpected case
            print(f"  Details: {e}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error: The script '{FETCH_SCRIPT}' failed.", file=sys.stderr)
        print(f"  --- STDOUT ---\n{e.stdout}", file=sys.stderr)
        print(f"  --- STDERR ---\n{e.stderr}", file=sys.stderr)
    except OSError as e:
        print(f"  ❌ Error during final cleanup of '{time_level_dir}': {e}", file=sys.stderr)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Sanitise, copy, and clean up specific meteor data directories.",
        epilog="Example: python3 false.py 20250816/005855 --dest-dir /tmp/my_wrongs"
    )
    # Argument parsing is unchanged
    parser.add_argument("directories", metavar="DIR", type=str, nargs='+', help="One or more directories to process.")
    parser.add_argument("--dest-dir", metavar="DIR_PATH", type=str, default="/var/www/html/wrongs/", help="Override the default destination directory (default: /var/www/html/wrongs/).")
    parser.add_argument("--dry-run", action="store_true", help="Print actions that would be taken without executing them.")
    parser.add_argument("--keep", action="store_true", help="Do not delete the source after copying, but list what would be deleted.")
    args = parser.parse_args()

    dest_base_path = Path(args.dest_dir)
    if not dest_base_path.exists() or not dest_base_path.is_dir():
        print(f"❌ Error: Destination directory '{dest_base_path}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("--- 💧 DRY RUN MODE ENABLED 💧 ---")
        print("No files will be copied or deleted.")
        print("-" * 35)

    for dir_arg in args.directories:
        if not sanitize_directory(dir_arg):
            print(f"⚠️ Warning: Discarding invalid directory format: '{dir_arg}'", file=sys.stderr)
            continue

        source_dir = Path(dir_arg)
        if not source_dir.is_dir():
            source_dir = ALT_SOURCE_PREFIX / dir_arg
        
        # In dry_run, we don't need the source to exist to show what would happen
        if not source_dir.is_dir() and not args.dry_run:
            print(f"⚠️ Warning: Source not found in current path or '{ALT_SOURCE_PREFIX}'. Skipping '{dir_arg}'", file=sys.stderr)
            continue
        
        dest_dir = dest_base_path / dir_arg
        process_directory(source_dir, dest_dir, dir_arg, args.dry_run, args.keep)

    print("\n✅ Script finished.")

if __name__ == "__main__":
    main()

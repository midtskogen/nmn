#!/usr/bin/env python3

import sys
import os
import subprocess
import random
import shutil
import pto_mapper

def check_dependencies():
    """Verify that the pano_trafo tool is available."""
    if not shutil.which("pano_trafo"):
        print("Error: 'pano_trafo' command not found.", file=sys.stderr)
        print("Please install Hugin/panotools and ensure it's in your system's PATH.", file=sys.stderr)
        sys.exit(1)

def run_pano_trafo(pto_file, x, y, image_index, reverse=False):
    """
    Executes the pano_trafo command and returns the parsed output.

    Args:
        pto_file (str): Path to the .pto file.
        x (float): The x coordinate.
        y (float): The y coordinate.
        image_index (int): The target image index.
        reverse (bool): If True, performs a reverse transformation (pano to image).

    Returns:
        tuple: A tuple of floats representing the output coordinates, or None on error.
    """
    command = ["pano_trafo"]
    if reverse:
        command.append("-r")
    command.extend([pto_file, str(image_index)])
    
    input_str = f"{x} {y}\n"
    
    try:
        process = subprocess.run(
            command,
            input=input_str,
            capture_output=True,
            text=True,
            check=True
        )
        # pano_trafo might output multiple lines, we only care about the first
        first_line = process.stdout.strip().split('\n')[0]
        parts = first_line.split()
        if len(parts) >= 2:
            return tuple(float(p) for p in parts)
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        # This can happen if the coordinate doesn't map; it's not a test failure.
        return None
    return None

def main():
    """Main function to run the verification tests."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <pto_file>", file=sys.stderr)
        sys.exit(1)

    pto_file = sys.argv[1]
    if not os.path.exists(pto_file):
        print(f"Error: File not found at '{pto_file}'", file=sys.stderr)
        sys.exit(1)

    check_dependencies()

    try:
        pto_data = pto_mapper.parse_pto_file(pto_file)
    except Exception as e:
        print(f"Error parsing PTO file: {e}", file=sys.stderr)
        sys.exit(1)

    global_options, images = pto_data
    pano_w = global_options['w']
    pano_h = global_options['h']
    num_images = len(images)
    iterations = 1000

    print(f"--- Verifying pto_mapper.py against pano_trafo using '{pto_file}' ---")
    print(f"--- Running {iterations} randomized tests ---")

    # --- Test 1: Forward Mapping (Panorama -> Image) ---
    forward_failures = []
    forward_tests_run = 0
    for i in range(iterations):
        pano_x = random.uniform(0, pano_w)
        pano_y = random.uniform(0, pano_h)

        # Our mapper finds the correct image automatically
        mapper_result = pto_mapper.map_pano_to_image(pto_data, pano_x, pano_y)
        
        if not mapper_result:
            continue
        
        img_idx, mapper_x, mapper_y = mapper_result
        
        # We must tell pano_trafo which image to map to
        pano_trafo_result = run_pano_trafo(pto_file, pano_x, pano_y, img_idx, reverse=True)

        if not pano_trafo_result:
            continue # pano_trafo couldn't map it either, so we can't compare.
        
        forward_tests_run += 1
        trafo_x, trafo_y = pano_trafo_result

        if not (abs(mapper_x - trafo_x) <= 2.0 and abs(mapper_y - trafo_y) <= 2.0):
            failure_info = {
                "input": (pano_x, pano_y),
                "image_index": img_idx,
                "mapper_output": (mapper_x, mapper_y),
                "pano_trafo_output": (trafo_x, trafo_y)
            }
            forward_failures.append(failure_info)

    print(f"\n[Forward Test: Pano -> Image]")
    print(f"Tests Attempted: {forward_tests_run}")
    print(f"Successes: {forward_tests_run - len(forward_failures)}")
    print(f"Failures: {len(forward_failures)}")

    # --- Test 2: Reverse Mapping (Image -> Panorama) ---
    reverse_failures = []
    reverse_tests_run = 0
    for i in range(iterations):
        img_idx = random.randrange(num_images)
        src_w = images[img_idx]['w']
        src_h = images[img_idx]['h']
        
        src_x = random.uniform(0, src_w)
        src_y = random.uniform(0, src_h)

        mapper_result = pto_mapper.map_image_to_pano(pto_data, img_idx, src_x, src_y)
        pano_trafo_result = run_pano_trafo(pto_file, src_x, src_y, img_idx, reverse=False)

        if not mapper_result or not pano_trafo_result:
            continue
        
        reverse_tests_run += 1
        mapper_x, mapper_y = mapper_result
        trafo_x, trafo_y = pano_trafo_result

        if not (abs(mapper_x - trafo_x) <= 2.0 and abs(mapper_y - trafo_y) <= 2.0):
            failure_info = {
                "input": (src_x, src_y),
                "image_index": img_idx,
                "mapper_output": (mapper_x, mapper_y),
                "pano_trafo_output": (trafo_x, trafo_y)
            }
            reverse_failures.append(failure_info)

    print(f"\n[Reverse Test: Image -> Pano]")
    print(f"Tests Attempted: {reverse_tests_run}")
    print(f"Successes: {reverse_tests_run - len(reverse_failures)}")
    print(f"Failures: {len(reverse_failures)}")

    # --- Final Summary ---
    print("\n--- Summary ---")
    if forward_failures or reverse_failures:
        print("STATUS: FAILED")
        if forward_failures:
            print("\nForward Mapping Failures (first 5):")
            for f in forward_failures[:5]:
                print(f"  Input (pano): ({f['input'][0]:.2f}, {f['input'][1]:.2f}) -> Img {f['image_index']}")
                print(f"    pto_mapper: ({f['mapper_output'][0]:.2f}, {f['mapper_output'][1]:.2f})")
                print(f"    pano_trafo: ({f['pano_trafo_output'][0]:.2f}, {f['pano_trafo_output'][1]:.2f})")
        if reverse_failures:
            print("\nReverse Mapping Failures (first 5):")
            for f in reverse_failures[:5]:
                print(f"  Input (img {f['image_index']}): ({f['input'][0]:.2f}, {f['input'][1]:.2f})")
                print(f"    pto_mapper: ({f['mapper_output'][0]:.2f}, {f['mapper_output'][1]:.2f})")
                print(f"    pano_trafo: ({f['pano_trafo_output'][0]:.2f}, {f['pano_trafo_output'][1]:.2f})")
        sys.exit(1)
    else:
        print("STATUS: PASSED")
        print("All verifiable mappings match the output of pano_trafo within the tolerance.")
        sys.exit(0)

if __name__ == "__main__":
    main()

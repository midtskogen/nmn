#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm
import os
import typing
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

def get_video_properties(video_path: str) -> typing.Tuple[typing.Optional[cv2.VideoCapture], ...]:
    """Safely opens a video and returns its properties."""
    if not os.path.exists(video_path):
        tqdm.write(f"Warning: Video file not found at '{video_path}'. Skipping.", file=sys.stderr)
        return None, None, None, None, None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tqdm.write(f"Warning: Could not open video file at '{video_path}'. Skipping.", file=sys.stderr)
        return None, None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not all([width, height, fps > 0, frame_count > 0]):
        tqdm.write(f"Warning: Could not read valid properties from '{video_path}'. Skipping.", file=sys.stderr)
        cap.release()
        return None, None, None, None, None

    return cap, width, height, fps, frame_count

def video_stack_worker(task: dict) -> typing.Optional[typing.Tuple[np.ndarray, dict]]:
    """
    A worker function that creates a stacked image for a single video file task.
    This function is designed to be run in a separate thread.
    """
    cap, width, height, _, _ = get_video_properties(task["path"])
    if not cap: return None

    individual_yuv_stack = np.zeros((height, width, 3), dtype=np.uint8)
    individual_yuv_stack[:, :, 1:] = 128
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, task["start_frame"])
    
    for _ in range(task["frame_count"]):
        ret, frame = cap.read()
        if not ret: break
        
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Update the stack for this video
        update_mask = yuv_frame[:, :, 0] > individual_yuv_stack[:, :, 0]
        individual_yuv_stack[update_mask] = yuv_frame[update_mask]
        
    cap.release()
    return individual_yuv_stack, task

def stack_video_frames(video_paths: list, output_path: str, start_seconds: float, duration_seconds: typing.Optional[float], denoise: bool, denoise_strength: float, quality: int, num_threads: int, individual: bool):
    """
    Reads and stacks video files in parallel, saving individual and/or a combined result.
    """
    processing_plan = []
    
    # --- Planning Phase ---
    print("Pre-scanning video files to create processing plan...")
    time_cursor = 0.0
    for path in video_paths:
        cap, _, _, fps, frame_count = get_video_properties(path)
        if not cap: continue
        video_duration = frame_count / fps
        end_boundary = float('inf') if duration_seconds is None else start_seconds + duration_seconds
        effective_start_time = max(time_cursor, start_seconds)
        effective_end_time = min(time_cursor + video_duration, end_boundary)

        if effective_end_time > effective_start_time:
            start_frame = int((effective_start_time - time_cursor) * fps)
            frames_to_process_in_video = int((effective_end_time - effective_start_time) * fps)
            if frames_to_process_in_video > 0:
                processing_plan.append({"path": path, "start_frame": start_frame, "frame_count": frames_to_process_in_video})
        time_cursor += video_duration
        cap.release()

    if not processing_plan:
        print("Error: No frames found to process with the given settings.", file=sys.stderr)
        sys.exit(1)

    # --- Execution Phase ---
    print(f"Ready to process {len(processing_plan)} video file(s) using up to {num_threads} threads.")
    total_yuv_stack = None
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(video_stack_worker, task) for task in processing_plan]
        
        with tqdm(total=len(processing_plan), desc="Processing Files", unit="file") as pbar:
            for future in futures:
                result = future.result()
                pbar.update(1)
                if result is None: continue
                
                individual_stack, task = result
                
                if total_yuv_stack is None:
                    total_yuv_stack = np.copy(individual_stack)
                else:
                    update_mask = individual_stack[:, :, 0] > total_yuv_stack[:, :, 0]
                    total_yuv_stack[update_mask] = individual_stack[update_mask]

                if individual:
                    base_name = os.path.splitext(os.path.basename(task['path']))[0]
                    individual_output_path = f"{base_name}_max.jpg"
                    bgr_image = cv2.cvtColor(individual_stack, cv2.COLOR_YUV2BGR)
                    finalize_and_save(bgr_image, individual_output_path, denoise, denoise_strength, quality, pbar=pbar)

    if total_yuv_stack is None:
        print("\nNo frames were stacked. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print("\nFinalizing combined stack of all videos...")
    final_bgr_image = cv2.cvtColor(total_yuv_stack, cv2.COLOR_YUV2BGR)
    finalize_and_save(final_bgr_image, output_path, denoise, denoise_strength, quality)

def stack_raw_frames(width: int, height: int, output_path: str, start_seconds: float, duration_seconds: typing.Optional[float], fps: float, denoise: bool, denoise_strength: float, quality: int):
    """
    Reads raw YUV420 data from stdin, stacks frames, and saves the result.
    """
    print(f"Processing raw YUV420 stream at {width}x{height}, {fps} FPS.")
    
    y_size, uv_size = width * height, (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size

    luma_stack = np.zeros((height, width), dtype=np.uint8)
    chroma_u_stack = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    chroma_v_stack = np.full((height // 2, width // 2), 128, dtype=np.uint8)

    frames_to_skip = int(start_seconds * fps)
    if frames_to_skip > 0:
        print(f"Skipping {frames_to_skip} frames ({start_seconds}s)...")
        for _ in range(frames_to_skip):
            if not sys.stdin.buffer.read(frame_size): break

    total_frames_to_process = int(duration_seconds * fps) if duration_seconds is not None else None
    with tqdm(total=total_frames_to_process, desc="Stacking Raw Frames", unit="frame") as pbar:
        while True:
            if total_frames_to_process is not None and pbar.n >= total_frames_to_process: break
            frame_data = sys.stdin.buffer.read(frame_size)
            if len(frame_data) < frame_size: break

            y_plane = np.frombuffer(frame_data[:y_size], dtype=np.uint8).reshape((height, width))
            u_plane = np.frombuffer(frame_data[y_size:y_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
            v_plane = np.frombuffer(frame_data[y_size + uv_size:frame_size], dtype=np.uint8).reshape((height // 2, width // 2))

            update_mask = y_plane > luma_stack
            np.maximum(luma_stack, y_plane, out=luma_stack)
            chroma_mask = cv2.resize(update_mask.astype(np.uint8), (width // 2, height // 2), interpolation=cv2.INTER_NEAREST).astype(bool)
            chroma_u_stack[chroma_mask] = u_plane[chroma_mask]
            chroma_v_stack[chroma_mask] = v_plane[chroma_mask]
            pbar.update(1)

    if pbar.n == 0:
        print("\nNo frames were stacked. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print("\nReconstructing final image from YUV planes...")
    final_yuv420_data = np.concatenate([luma_stack.ravel(), chroma_u_stack.ravel(), chroma_v_stack.ravel()])
    yuv_image_i420 = final_yuv420_data.reshape((int(height * 1.5), width))
    final_bgr_image = cv2.cvtColor(yuv_image_i420, cv2.COLOR_YUV2BGR_I420)
    finalize_and_save(final_bgr_image, output_path, denoise, denoise_strength, quality, pbar=pbar)

def finalize_and_save(bgr_image, output_path: str, denoise: bool, denoise_strength: float, quality: int, pbar: typing.Optional[tqdm] = None):
    """Denoises (if enabled) and saves the final BGR image, logging with tqdm if provided."""
    log = tqdm.write if pbar else print
    
    final_image = bgr_image
    if denoise:
        log(f"Denoising '{output_path}' with strength {denoise_strength}...")
        final_image = cv2.fastNlMeansDenoisingColored(bgr_image, None, h=denoise_strength, hColor=denoise_strength, templateWindowSize=7, searchWindowSize=21)
    
    log(f"Saving result to '{output_path}'...")
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(output_path, final_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(output_path, final_image)

def main():
    parser = argparse.ArgumentParser(description="Stack video frames to simulate a long exposure.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("video_files", nargs='*', help="Path(s) to input video file(s). Ignored if --raw is used.")
    parser.add_argument("-o", "--output", default="max.jpg", help="Path for the final combined output image. Format is determined by extension. Default: max.jpg")
    parser.add_argument("-s", "--start", type=float, default=0.0, help="Start position in seconds.")
    parser.add_argument("-t", "--duration", type=float, default=None, help="Total duration in seconds to stack.")
    parser.add_argument("--individual", action="store_true", help="Save a separate stacked image for each input file.")
    parser.add_argument("--denoise", action="store_true", help="Enable a denoising filter on the final image(s).")
    parser.add_argument("--denoise-strength", type=float, default=10.0, help="Strength of the denoising filter (default: 10.0).")
    parser.add_argument("--quality", type=int, default=95, choices=range(1, 101), metavar="{1-100}", help="Quality for JPEG output (1-100, default: 95).")
    parser.add_argument("--raw", nargs=2, metavar=('W', 'H'), type=int, help="Process raw YUV420 from stdin with specified Width and Height.")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for raw stream timing (default: 25.0).")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="Number of threads for parallel video decoding (default: all available cores).")
    args = parser.parse_args()

    if args.raw and args.individual:
        parser.error("The --raw and --individual options cannot be used together.")

    if args.raw:
        if args.start > 0 or args.duration is not None:
            print(f"Info: Using FPS of {args.fps} for time calculations in raw mode.", file=sys.stderr)
        width, height = args.raw
        stack_raw_frames(width, height, args.output, args.start, args.duration, args.fps, args.denoise, args.denoise_strength, args.quality)
    else:
        if not args.video_files:
            parser.error("Missing input. Provide video file(s) or use the --raw option.")
        stack_video_frames(args.video_files, args.output, args.start, args.duration, args.denoise, args.denoise_strength, args.quality, args.threads, args.individual)

if __name__ == "__main__":
    main()

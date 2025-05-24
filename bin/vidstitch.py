#!/usr/bin/python3

import os
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil
import glob

def get_video_start_time(video_file):
    """Uses ffprobe to get the start time of the video."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=start_time',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout.strip()) if result.returncode == 0 else 0.0

def extract_frames(video_file, start_time, output_dir, camera_id):
    """Extracts frames from a video starting from a specific time."""
    output_pattern = os.path.join(output_dir, f"frame{camera_id}_%06d.jpg")
    cmd = [
        'ffmpeg', '-v', 'error',
#        '-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi',
        '-ss', str(start_time), '-i', video_file,
#        '-vf', 'hwdownload,format=nv12',
        output_pattern
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)

def stitch_frames_with_nona(pto_file, temp_dir, frame_num):
    """Uses nona to stitch frames from different cameras for a specific frame."""
    frame_files = [os.path.join(temp_dir, f"frame{i+1}_{frame_num:06d}.jpg") for i in range(7)]
    output_stitched = os.path.join(temp_dir, f"stitched_{frame_num:06d}_")
    cmd = ['nona', '-o', output_stitched, pto_file] + frame_files
    print(" ".join(cmd))
    subprocess.run(cmd)
    output_file = os.path.join(temp_dir, f"stitched_{frame_num:06d}.jpg")
    if frame_num == 1:
        cmd = ['enblend', '--save-masks', '-o', output_file] + glob.glob(output_stitched + "*")
    else:
        cmd = ['multiblend', '--load-masks', '-o', output_file] + glob.glob(output_stitched + "*")
    print(" ".join(cmd))
    subprocess.run(cmd)
#    cmd = ['convert', '-crop', '2895x2895+1112+1112', output_file, output_file + ".jpg"]
#    subprocess.run(cmd)
#    cmd = ['mv', output_file + ".jpg", output_file] 
#    subprocess.run(cmd)
    cmd = ['rm'] + glob.glob(output_stitched + "*") + frame_files
    print(" ".join(cmd))
    subprocess.run(cmd)
    return output_file

def encode_stitched_frames_to_video(temp_dir, output_video_file):
    """Encodes all stitched frames back into a video."""
    frame_pattern = os.path.join(temp_dir, 'stitched_%06d.jpg')
    cmd = [
        'ffmpeg', '-hwaccel', 'vaapi', '-framerate 25', ',-i', 'stitched_%06d.jpg',
        '-vf', 'swapuv,format=nv12,hwupload,scale_vaapi=w=trunc(iw/16)*16:h=trunc(ih/16)*16',
        '-c:v', 'h264_vaapi', output_video_file
    ]
    subprocess.run(cmd)

def main(video_path, pto_file):
    # Define paths
    base_video_path = Path(video_path)
    cameras = [f"/meteor/cam{i}/{base_video_path}" for i in range(1, 8)]
    
    # Get the start times of all videos
    start_times = {cam: get_video_start_time(cam) for cam in cameras}
    max_start_time = max(start_times.values())

    # Create temporary working directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Extract frames from each video synchronizing based on the max start time
        for i, cam in enumerate(cameras):
            time_diff = max_start_time - start_times[cam]
            extract_frames(cam, time_diff, temp_dir, i + 1)

        # Process frames and stitch them
        frame_num = 1
        stitched_files = []
        while True:
            stitched_frame = stitch_frames_with_nona(pto_file, temp_dir, frame_num)
            if stitched_frame:
                stitched_files.append(stitched_frame)
            else:
                break
            frame_num += 1

        # Encode the final stitched frames into a video
        output_video_file = f"stitched_{base_video_path.stem}.mp4"
        encode_stitched_frames_to_video(temp_dir, output_video_file)

        print(f"Stitched video created: {output_video_file}")

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stitch_videos.py <video_path> <pto_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    pto_file = sys.argv[2]
    main(video_path, pto_file)

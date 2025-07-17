#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tkinter as tk
import argparse
import ffmpeg
import tempfile
import glob
import os
import configparser
import ephem
import math
import io
import wand
import hsi
import sys
import re
import subprocess
from datetime import datetime, timedelta, UTC
from tkinter import ttk
from PIL import Image, ImageTk, ImageEnhance
from brightstar import brightstar
from recalibrate import recalibrate
from timestamp import get_timestamp

class RecalibrateDialog:
    def __init__(self, parent, image, pano, zoom):
        self.parent = parent
        self.image = image
        self.pano = pano
        self.zoom = zoom
        self.parent.title("Recalibrate")

        # Initialize variables for sliders
        self.radius_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.IntVar(value=50)
        self.sigma_var = tk.IntVar(value=20)
        self.lens_optimize_var = tk.BooleanVar(value=True)

        # Create sliders
        self.create_slider("Radius", 0.1, 5.0, self.radius_var, row=0)
        self.create_slider("Blur", 1, 100, self.blur_var, row=1)
        self.create_slider("Sigma", 1, 100, self.sigma_var, row=2)

        self.pitch = img.getPitch()
        self.yaw = img.getYaw()
        self.roll = img.getRoll()
        self.hfov = img.getHFOV()
        self.raddist = img.getRadialDistortion()
        self.raddistcent = img.getRadialDistortionCenterShift()

        # Create button for Lens optimization
        lens_optimize_btn = tk.Checkbutton(self.parent, text="Optimize lens parameters", variable=self.lens_optimize_var)
        lens_optimize_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky='w')

        # Create Solve button
        solve_btn = tk.Button(self.parent, text="Solve", command=self.recal)
        solve_btn.grid(row=4, column=0, columnspan=2, pady=10, sticky='w')

        # Create Reset button
        close_btn = tk.Button(self.parent, text="Reset", command=self.reset)
        close_btn.grid(row=4, column=1, columnspan=2, pady=10, sticky='w')

        # Create Close button
        close_btn = tk.Button(self.parent, text="Close", command=self.parent.destroy)
        close_btn.grid(row=6, column=0, columnspan=2, pady=10, sticky='w')

        # Create labels for help text
        help_label = tk.Label(self.parent, font=("Helvetica", 14, "bold"), text="Fine-tuning the calibration")
        help_label.grid(row=0, column=3, sticky='w', padx=(10, 0))

        help_text = """
        The sliders controls the following values:

        - "Radius" the search mask in degrees to look for the star.
        - "Blur" controls the amount of blurring for the mask.
        - "Sigma" indicates how much noise should be assumed.
        - "Optimise lens parameters" for full recalibration, not only orientation.
        - Click "Solve" to perform the calibration.
        - Click "Reset" to go back to the original parameters.
        - The "Close" button closes the dialog.
        """
        help_text_label = tk.Label(self.parent, text=help_text, justify=tk.LEFT)
        help_text_label.grid(row=1, column=3, rowspan=5, sticky='w', padx=(10, 0))

        par_text = "Original parameters:\n  pitch=%.2f°\n  yaw=%.2f°\n  roll=%.2f°\n  hfov=%.2f°\n  radial=(%.3f, %.3f, %.3f)\n  radial shift=(%.1f, %.1f)" % (self.pitch, self.yaw, self.roll, self.hfov, self.raddist[0], self.raddist[1], self.raddist[2], self.raddistcent.x, self.raddistcent.y)
        par_text_label = tk.Label(self.parent, text=par_text, justify=tk.LEFT)
        par_text_label.grid(row=7, column=0, sticky='w', padx=(10, 0))

    def create_slider(self, label_text, min_val, max_val, variable, row):
        label = tk.Label(self.parent, text=label_text)
        label.grid(row=row, column=0, sticky='w')

        slider_frame = ttk.Frame(self.parent)
        slider_frame.grid(row=row, column=1, sticky='w')

        slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=variable, command=lambda val, lbl=label_text: self.update_slider_label(val, lbl))
        slider.pack(pady=5)

        # Display the selected value
        value_label = tk.Label(slider_frame, text=f"{label_text}: {variable.get()}")
        value_label.pack()

    def update_slider_label(self, value, label_text):
        # Find the label associated with the slider
        for child in self.parent.winfo_children():
            if isinstance(child, ttk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, tk.Label) and subchild.cget("text").startswith(label_text):
                        # Update the label text with the selected value
                        subchild.config(text=f"{label_text}: {float(value):.2f}")

    def reset(self):
        img.setPitch(self.pitch)
        img.setYaw(self.yaw)
        img.setRoll(self.roll)
        img.setHFOV(self.hfov)
        img.setRadialDistortion(self.raddist)
        img.setRadialDistortionCenterShift(self.raddistcent)
        self.zoom.show_image()

    def recal(self):
        stream = io.BytesIO()
        self.image.save(stream, format='PNG')
        stream.seek(0)
        old_lens = tempfile.NamedTemporaryFile(delete=True, dir="/tmp").name
        try:
            pano.WritePTOFile(old_lens)
        except:
            pano.writeData(hsi.ofstream(old_lens))
        new_lens = tempfile.NamedTemporaryFile(delete=True, dir="/tmp").name
        recalibrate(starttime, old_lens, wand.image.Image(blob=stream.read()), new_lens, pos, image=0, radius=self.radius_var.get(), lensopt=self.lens_optimize_var.get(), faintest=4, brightest=-5, objects=500, blur=self.blur_var.get(), verbose=True, sigma=self.sigma_var.get())
        pano2 = hsi.Panorama()
        try:
            pano2.ReadPTOFile(new_lens)
        except:
            pano2.readData(hsi.ifstream(new_lens))
        img2 = pano2.getImage(args.image)
        img.setPitch(img2.getPitch())
        img.setYaw(img2.getYaw())
        img.setRoll(img2.getRoll())
        img.setHFOV(img2.getHFOV())
        img.setRadialDistortion(img2.getRadialDistortion())
        img.setRadialDistortionCenterShift(img2.getRadialDistortionCenterShift())
        self.zoom.show_image()

parser = argparse.ArgumentParser(description='Click on images to find coordinates.')

parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)
parser.add_argument('-f', '--fps', dest='fps', help='frames per second (default: 10 or extracted from images)', default=10, type=int)
parser.add_argument('-n', '--name', dest='name', help='station name (default: NUL or extracted from station config)', default="NUL", type=str)
parser.add_argument('-s', '--station', dest='station', help='station config file', type=str)
parser.add_argument('-d', '--date', dest='start', help='start time (default: extracted from images))', type=str)
parser.add_argument(action='store', dest='ptofile', help='input .pto file')
parser.add_argument(action='store', nargs="*", dest='imgfiles', help='input image or video files (.mp4 or image files)')
if __name__ == '__main__':
    args = parser.parse_args()

pos = None
if args.station != None:
    pos = ephem.Observer()
    config = configparser.ConfigParser()
    config.read(args.station)
    pos.lat = config.get('astronomy', 'latitude')
    pos.lon = config.get('astronomy', 'longitude')
    pos.elevation = float(config.get('astronomy', 'elevation'))
    pos.temp = float(config.get('astronomy', 'temperature'))
    pos.pressure = float(config.get('astronomy', 'pressure'))
    args.name = config.get('station', 'code')


def read_frames(filename, directory):
    """Decodes a video file into individual frames, showing a progress bar."""
    try:
        probe = ffmpeg.probe(filename)
        total_frames = int(next(s['nb_frames'] for s in probe['streams'] if s['codec_type'] == 'video'))
    except (ffmpeg.Error, StopIteration, KeyError, ValueError):
        total_frames = None

    print(f"Decoding video {filename}...")

    # Build the ffmpeg command arguments using ffmpeg-python
    stream = ffmpeg.input(filename).output(f'{directory}/%04d.tif', format='image2', vsync=0).overwrite_output()
    args = stream.get_args()
    
    # Add progress reporting if we have the total frame count
    if total_frames:
        # Use a list to prepend arguments
        final_args = ['ffmpeg', '-progress', 'pipe:1'] + args
    else:
        final_args = ['ffmpeg'] + args

    proc = subprocess.Popen(final_args, stdout=subprocess.PIPE if total_frames else None, stderr=subprocess.PIPE)

    if not total_frames:
        # If we don't know the total frames, we can't show a progress bar. Just wait.
        _, err = proc.communicate()
        if proc.returncode != 0:
            raise ffmpeg.Error('ffmpeg', None, err)
        return

    # If we have total_frames, parse stdout for progress
    bar_length = 40
    for line in iter(proc.stdout.readline, b''):
        line_str = line.decode('utf-8')
        match = re.search(r'frame=\s*(\d+)', line_str)
        if match:
            current_frame = int(match.group(1))
            progress = current_frame / total_frames
            block = int(round(bar_length * progress))
            text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {current_frame}/{total_frames} ({progress*100:.0f}%)"
            sys.stdout.write(text)
            sys.stdout.flush()

    # Finalize the progress bar
    text = f"\rProgress: [{'#' * bar_length}] {total_frames}/{total_frames} (100%)"
    sys.stdout.write(text)
    sys.stdout.write("\n\n")

    err = proc.stderr.read()
    if proc.wait() != 0:
        raise ffmpeg.Error('ffmpeg', None, err)


#def read_timestamps(files):
#    for f in files:
#        ts = subprocess.run(["timestamp", f], cwd=str(pathlib.Path.home()) + "/bin", stdout=subprocess.PIPE, text=True)
#        timestamps.append(float(ts.stdout.rstrip().lstrip()))
#    return timestamps

def timestamp(timestamps, files, i):
    """
    Gets the timestamp for the file at index i, using a cache.
    It calls get_timestamp, first normally, then in robust mode on failure.
    """
    if i < len(timestamps) and timestamps[i] is not None:
        return timestamps[i]

    # First attempt: Call get_timestamp without robust mode for speed.
    datetime_obj = get_timestamp(files[i])

    # If the first attempt fails, try again with robust mode.
    if not datetime_obj:
        datetime_obj = get_timestamp(files[i], robust=True)

    # Proceed with the result from the attempts.
    if datetime_obj:
        ts = datetime_obj.timestamp()
        timestamps[i] = ts
        return ts
    else:
        # If a timestamp can't be read even with robust mode, fall back.
        if i > 0:
            # Assume the timestamp is the same as the previous frame.
            timestamps[i] = timestamps[i-1]
            return timestamps[i-1]
        else:
            # If the very first frame is unreadable, we cannot proceed.
            sys.stderr.write(f"Error: Could not extract timestamp from the first image: {files[i]}\n")
            sys.stderr.write("Please provide a start time using the -d argument (e.g., -d '2023-10-27 10:30:00.000')\n")
            sys.exit(1)

def interpolate_timestamps(timestamps):
    """Performs interpolation on a pre-populated list of timestamps."""
    if not timestamps:
        return []
    
    interpolated_stamps = list(timestamps) # Make a copy

    fractions = []
    prev = interpolated_stamps[0]

    i = 1
    previ = -1
    while i < len(interpolated_stamps):
        if interpolated_stamps[i] == prev:
            while i < len(interpolated_stamps) and interpolated_stamps[i] == prev:
                i += 1
            fractions.append(i - previ)
            previ = i
            if i < len(interpolated_stamps):
                prev = interpolated_stamps[i]
        else:
            previ += 1
            while interpolated_stamps[i] != prev:
                i -= 1
            fractions.append(i + 2 - previ)
            previ = i + 1
            if i + 1 < len(interpolated_stamps):
                prev = interpolated_stamps[i + 1]
            else:
                # Reached the end, break the loop
                break
        
        if not fractions or fractions[-1] == 0:
            # Avoid infinite loop if fraction logic fails
            i+=1
            continue
            
        i += fractions[-1]

    fractions.append(len(interpolated_stamps) - previ)
    if len(fractions) > 1:
        fractions[-1] = fractions[-2]

    prev = interpolated_stamps[0]
    if not fractions or len(fractions) < 2 or fractions[1] == 0:
         f = 0
    else:
        f = 1 - fractions[0] / fractions[1]

    fi = 0
    if not fractions or fractions[0] == 0:
        pass
    else:
        fractions[0] = fractions[1]
    
    for i in range(1, len(interpolated_stamps)):
        if interpolated_stamps[i] is None:
            interpolated_stamps[i] = interpolated_stamps[i - 1]
    
    for i in range(len(interpolated_stamps)):
        if interpolated_stamps[i] == prev:
            if fi < len(fractions) and fractions[fi] > 0:
                f = f + 1.0 / fractions[fi]
            interpolated_stamps[i] += round(f, 2)
        else:
            prev = interpolated_stamps[i]
            f = 0.0
            fi += 1

    return interpolated_stamps

def midpoint(az1, alt1, az2, alt2):
    x1 = math.radians(az1)
    x2 = math.radians(az2)
    y1 = math.radians(alt1)
    y2 = math.radians(alt2)
    Bx = math.cos(y2) * math.cos(x2-x1)
    By = math.cos(y2) * math.sin(x2-x1)
    y3 = math.atan2(math.sin(y1) + math.sin(y2),
                    math.sqrt( (math.cos(y1)+Bx)*(math.cos(y1)+Bx) + By*By ) )
    x3 = x1 + math.atan2(By, math.cos(y1) + Bx)
    a = math.sin((y2-y1)/2) * math.sin((y2-y1)/2) + math.cos(y1) * math.cos(y2) * math.sin((x2-x1)/2) * math.sin((x2-x1)/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return math.degrees(x3), math.degrees(y3), math.degrees(c)

# Pan/zoom from https://stackoverflow.com/questions/41656176/tkinter-canvas-zoom-move-pan
class Zoom_Advanced(ttk.Frame):
    ''' Advanced zoom of the image '''
    def __init__(self, mainframe, files, timestamps):
        ''' Initialize the main Frame '''
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Zoom with mouse wheel')
        # Create canvas and put image on it

        self.overlay = []
        self.canvas = tk.Canvas(self.master, highlightthickness=0, cursor="draft_small", bg="black")
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        self.canvas.bind('<Left>', self.left_key)
        self.canvas.bind('<Right>', self.right_key)
        self.canvas.bind('<Prior>', self.page_up)
        self.canvas.bind('<Next>', self.page_down)
        self.canvas.bind('<Key>', self.key)
        self.canvas.bind('<Control-p>', self.ctrl_p)
        self.canvas.bind('<Control-P>', self.ctrl_P)
        self.canvas.bind('<Control-y>', self.ctrl_y)
        self.canvas.bind('<Control-Y>', self.ctrl_Y)
        self.canvas.bind('<Control-r>', self.ctrl_r)
        self.canvas.bind('<Control-R>', self.ctrl_R)
        self.canvas.bind('<Button-3>', self.click)
        self.canvas.bind("<Motion>", self.moved)
        self.mousepos = "\n"
        self.canvas.focus_set()
        self.files = files
        self.timestamps = timestamps
        self.num = 0
        self.image = Image.open(files[0])  # open image
        self.width, self.height = self.image.size
        # print(get_timestamp(self.image))
        self.imscale = 1.0  # scale for the canvas image
        self.delta = 1.05  # zoom magnitude
        self.contrast = 1
        self.brightness = 1
        self.color = 1
        self.sharpness = 1
        self.show_text = 0
        self.show_info = 0
        self.boost = 1
        self.offsetx = 0
        self.offsety = 0
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        self.show_image()
        
    def moved(self, event):
        x, y = event.x / self.imscale + self.x, event.y / self.imscale + self.y
        x += self.offsetx
        y += self.offsety
        inv.transformImgCoord(dst, hsi.FDiff2D(x, y))
        ra, dec = pos.radec_of(str((dst.x / 100) % 360), str(90 - (dst.y / 100)))
        self.mousepos="\n  cursor pos = %.2f° az, %.2f° alt / %s ra %.2f dec" % ((dst.x / 100) % 360, 90 - (dst.y / 100),
                                                                                 str(ra), math.degrees(float(repr(dec))))
        self.show_image()

    def ctrl_p(self, event):
        hsi.RotatePanorama(pano, 0, 0.01 * self.boost, 0).run()
        inv.createInvTransform(img, pano.getOptions())
        self.show_image()

    def ctrl_P(self, event):
        hsi.RotatePanorama(pano, 0, -0.01 * self.boost, 0).run()
        inv.createInvTransform(img, pano.getOptions())
        self.show_image()

    def ctrl_y(self, event):
        hsi.RotatePanorama(pano, 0, 0, 0.01 * self.boost).run()
        inv.createInvTransform(img, pano.getOptions())
        self.show_image()

    def ctrl_Y(self, event):
        hsi.RotatePanorama(pano, 0, 0, -0.01 * self.boost).run()
        inv.createInvTransform(img, pano.getOptions())
        self.show_image()

    def ctrl_r(self, event):
        hsi.RotatePanorama(pano, 0.01 * self.boost, 0, 0).run()
        inv.createInvTransform(img, pano.getOptions())
        self.show_image()

    def ctrl_R(self, event):
        hsi.RotatePanorama(pano, -0.01 * self.boost, 0, 0).run()
        inv.createInvTransform(img, pano.getOptions())
        self.show_image()

    def key(self, event):
        if event.char == 'p':
            img.setPitch(img.getPitch() + 0.01 * self.boost)
        if event.char == 'P':
            img.setPitch(img.getPitch() - 0.01 * self.boost)
        if event.char == 'y':
            img.setYaw(img.getYaw() + 0.01 * self.boost)
        if event.char == 'Y':
            img.setYaw(img.getYaw() - 0.01 * self.boost)
        if event.char == 'r':
            img.setRoll(img.getRoll() + 0.01 * self.boost)
        if event.char == 'R':
            img.setRoll(img.getRoll() - 0.01 * self.boost)
        if event.char == 'z':
            img.setHFOV(img.getHFOV() + 0.01 * self.boost)
        if event.char == 'Z':
            img.setHFOV(img.getHFOV() - 0.01 * self.boost)
        if event.char == 'a':
            img.setRadialDistortion((img.getRadialDistortion()[0] + 0.001, img.getRadialDistortion()[1], img.getRadialDistortion()[2]))
        if event.char == 'A':
            img.setRadialDistortion((img.getRadialDistortion()[0] - 0.001, img.getRadialDistortion()[1], img.getRadialDistortion()[2]))
        if event.char == 'b':
            img.setRadialDistortion((img.getRadialDistortion()[0], img.getRadialDistortion()[1] + 0.001, img.getRadialDistortion()[2]))
        if event.char == 'B':
            img.setRadialDistortion((img.getRadialDistortion()[0], img.getRadialDistortion()[1] - 0.001, img.getRadialDistortion()[2]))
        if event.char == 'c':
            img.setRadialDistortion((img.getRadialDistortion()[0], img.getRadialDistortion()[1], img.getRadialDistortion()[2] + 0.001))
        if event.char == 'C':
            img.setRadialDistortion((img.getRadialDistortion()[0], img.getRadialDistortion()[1], img.getRadialDistortion()[2] - 0.001))
        if event.char == 'd':
            img.setRadialDistortionCenterShift(hsi.FDiff2D(img.getRadialDistortionCenterShift().x + 0.5, img.getRadialDistortionCenterShift().y))
        if event.char == 'D':
            img.setRadialDistortionCenterShift(hsi.FDiff2D(img.getRadialDistortionCenterShift().x - 0.5, img.getRadialDistortionCenterShift().y))
        if event.char == 'e':
            img.setRadialDistortionCenterShift(hsi.FDiff2D(img.getRadialDistortionCenterShift().x, img.getRadialDistortionCenterShift().y + 0.5))
        if event.char == 'E':
            img.setRadialDistortionCenterShift(hsi.FDiff2D(img.getRadialDistortionCenterShift().x, img.getRadialDistortionCenterShift().y - 0.5))
        if event.char == '?':
            root = tk.Tk()
            dialog_window = RecalibrateDialog(root, self.image, pano, self)
            root.mainloop()
        if event.char == '*':
            pano2 = hsi.Panorama()
            try:
                pano2.ReadPTOFile(args.ptofile)
            except:
                pano2.readData(hsi.ifstream(args.ptofile))
            img2 = pano2.getImage(args.image)
            img.setPitch(img2.getPitch())
            img.setYaw(img2.getYaw())
            img.setRoll(img2.getRoll())
            img.setHFOV(img2.getHFOV())
            img.setRadialDistortion(img2.getRadialDistortion())
            img.setRadialDistortionCenterShift(img2.getRadialDistortionCenterShift())
        if event.char == 'l':
            try:
                pano.WritePTOFile(args.ptofile)
            except:
                pano.writeData(hsi.ofstream(args.ptofile))
        if event.char == 'S':
            with open("centroid.txt","w") as f:
                for l in centroid:
                    if l != None:
                        f.write(l + "\n")
        if event.char == 's':
            centroid2 = [l for l in centroid if l is not None]
            if not centroid2:
                print("No centroid data to save to event.txt")
                return
                
            with open("event.txt", "w") as f:
                print("[trail]", file=f)
                print(f"frames = {len(centroid2)}", file=f)
                print(f"duration = {round(float(centroid2[-1].split(' ')[1]) - float(centroid2[0].split(' ')[1]), 2)}", file=f)
                
                pos_str = " ".join([f"{p[0]},{p[1]}" for p in positions])
                print(f"positions = {pos_str}", file=f)

                coord_str = " ".join([f"{i.split(' ')[3]},{i.split(' ')[2]}" for i in centroid2])
                print(f"coordinates = {coord_str}", file=f)

                ts_str = " ".join([str(self.timestamps[int(i.split(' ')[0])]) for i in centroid2])
                print(f"timestamps = {ts_str}", file=f)
                
                midaz, midalt, arc = midpoint(float(centroid2[0].split(" ")[3]), float(centroid2[0].split(" ")[2]),
                                              float(centroid2[-1].split(" ")[3]), float(centroid2[-1].split(" ")[2]))
                print(f"midpoint = {midaz},{midalt}", file=f)
                print(f"arc = {arc}", file=f)
                
                print("brightness = " + " ".join(["0"] * len(centroid2)), file=f)
                print("frame_brightness = " + " ".join(["0"] * len(centroid2)), file=f)
                print("size = " + " ".join(["0"] * len(centroid2)), file=f)
                
                print("manual = 1", file=f)
                print(file=f)

                print("[video]", file=f)
                start_info = " ".join(centroid2[0].split(" ")[6:])
                start_ts = self.timestamps[int(centroid2[0].split(" ")[0])]
                print(f"start = {start_info} ({start_ts})", file=f)
                
                end_info = " ".join(centroid2[-1].split(" ")[6:])
                end_ts = self.timestamps[int(centroid2[-1].split(" ")[0])]
                print(f"end = {end_info} ({end_ts})", file=f)
                
                print(f"width = {width}", file=f)
                print(f"height = {height}", file=f)
            
            print("Saved event data to event.txt")
        if event.char == 'h':
            self.show_text ^= 1
        if event.char == 'i':
            self.show_info ^= 1
        if event.char == '!':
            self.boost = 100 if self.boost == 1 else 1
        if event.char == 'q':
            exit(0)
        if event.char == '1':
            self.contrast -= 0.1
        if event.char == '2':
            self.contrast += 0.1
        if event.char == '3':
            self.brightness -= 0.1
        if event.char == '4':
            self.brightness += 0.1
        if event.char == '5':
            self.color -= 0.1
        if event.char == '6':
            self.color += 0.1
        if event.char == '7':
            self.sharpness -= 0.2
        if event.char == '8':
            self.sharpness += 0.2
        if event.char == '0':
            self.contrast = self.brightness = self.color = self.sharpness = 1
        inv.createInvTransform(img, pano.getOptions())
        self.show_image()
            
    def left_key(self, event):
        if self.num > 0:
            self.num = self.num - 1
            self.image = Image.open(self.files[self.num])
            self.show_image()

    def right_key(self, event):
        if self.num < len(self.files) - 1:
            self.num += 1
            self.image = Image.open(self.files[self.num])
            self.show_image()

    def page_up(self, event):
        if self.num > 0:
            self.num = max(0, self.num - 10)
            self.image = Image.open(self.files[self.num])
            self.show_image()

    def page_down(self, event):
        if self.num < len(self.files) - 1:
            self.num = min(len(self.files) - 1, self.num + 10)
            self.image = Image.open(self.files[self.num])
            self.show_image()

    def scroll_y(self, *args, **kwargs):
        ''' Scroll canvas vertically and redraw the image '''
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def scroll_x(self, *args, **kwargs):
        ''' Scroll canvas horizontally and redraw the image '''
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        else: return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            if self.imscale / self.delta < 1:
                return
            self.imscale /= self.delta
            scale        /= self.delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale        *= self.delta

        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.show_image()

    def converted_image(self, img_a, contrast, brightness, color, sharpness):
        contrast_converter = ImageEnhance.Contrast(img_a)
        img_b = contrast_converter.enhance(contrast)
        brightness_converter = ImageEnhance.Brightness(img_b)
        img_c = brightness_converter.enhance(brightness)
        color_converter = ImageEnhance.Color(img_c)
        img_d = color_converter.enhance(color)
        sharpness_converter = ImageEnhance.Sharpness(img_d)
        img_final = sharpness_converter.enhance(sharpness)
        return img_final

    def show_image2(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            self.x = x1 / self.imscale
            self.y = y1 / self.imscale
            image = self.image.crop((x1 / self.imscale, y1 / self.imscale, x, y))
            imagetk = ImageTk.PhotoImage(image.resize((x2 - x1, y2 - y1)))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection
            
    def show_image(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region

        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        self.offsetx = min(bbox2[0] - bbox1[0], 0) / self.imscale
        self.offsety = min(bbox2[1] - bbox1[1], 0) / self.imscale
        try: self.timestamps
        except:
            self.timestamps = []
            self.timestamps.append(0)
            self.timestamps.append(0)
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            pos.date = datetime.fromtimestamp(self.timestamps[self.num], UTC)
            stars = brightstar(pano, args.image, pos, 6.5, -30, int(128*self.imscale))
            x = min(x2 / self.imscale, self.width)   # sometimes it is larger on 1 pixel...
            y = min(y2 / self.imscale, self.height)  # ...and sometimes not
            self.x = x1 / self.imscale
            self.y = y1 / self.imscale
            image = self.image.crop((self.x, self.y, x, y))
            image = self.converted_image(image, self.contrast, self.brightness, self.color, self.sharpness)
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)

            for c in self.overlay:
                self.canvas.delete(c)
            help_id = self.canvas.create_text(bbox2[0], bbox2[1], anchor="nw")
            ts = datetime.fromtimestamp(self.timestamps[self.num], UTC)
            self.canvas.itemconfig(help_id, text=
                                   "  time = " + ts.strftime("%Y-%m-%d %H:%M:%S.%f UTC") + " (%.2f)" % self.timestamps[self.num] +
                                   "\n  pitch=%.2f° yaw=%.2f° roll=%.2f° hfov=%.2f° radial=(%.3f, %.3f, %.3f) radial shift=(%.1f, %.1f)" % (img.getPitch(), img.getYaw(), img.getRoll(), img.getHFOV(), img.getRadialDistortion()[0], img.getRadialDistortion()[1], img.getRadialDistortion()[2], img.getRadialDistortionCenterShift().x, img.getRadialDistortionCenterShift().y) +
                                   self.mousepos +
                                   "\n  h = toggle help text",
                                   fill="yellow", font=("helvetica", 10))
            self.overlay.append(help_id)

            if self.show_text == 1:
                text_id = self.canvas.create_text(bbox2[0], bbox2[1], anchor="nw")
                self.canvas.itemconfig(text_id, text=
                                       "\n\n\n\n"
                                       "  q = quit\n"
                                       "  1 = decrease contrast\n"
                                       "  2 = increase contrast\n"
                                       "  3 = decrease brightness\n"
                                       "  4 = increase brightness\n"
                                       "  5 = decrease colour\n"
                                       "  6 = increase colour\n"
                                       "  7 = decrease sharpness\n"
                                       "  8 = increase sharpness\n"
                                       "  0 = reset enhancement\n"
                                       "  p = increase pitch by 0.01 degrees (ctrl-p = rotate)\n"
                                       "  P = decrease pitch by 0.01 degrees (ctrl-p = rotate)\n"
                                       "  y = increase yaw by 0.01 degrees (ctrl-y = rotate)\n"
                                       "  Y = decrease yaw by 0.01 degrees (ctrl-Y = rotate)\n"
                                       "  r = increase roll by 0.01 degrees (ctrl-r = rotate)\n"
                                       "  R = decrease roll by 0.01 degrees (ctrl-R = rotate)\n"
                                       "  a = increase radial distortion param 0 by 0.001\n"
                                       "  A = decrease radial distortion param 0 by 0.001\n"
                                       "  b = increase radial distortion param 1 by 0.001\n"
                                       "  B = decrease radial distortion param 1 by 0.001\n"
                                       "  c = increase radial distortion param 2 by 0.001\n"
                                       "  C = decrease radial distortion param 2 by 0.001\n"
                                       "  d = increase radial distortion shift x by 0.5\n"
                                       "  D = decrease radial distortion shift x by 0.5\n"
                                       "  e = increase radial distortion shift y by 0.5\n"
                                       "  E = decrease radial distortion shift y by 0.5\n"
                                       "  i = toggle star info\n"
                                       "  ! = toggle boost mode (100x)\n"
                                       "  * = reset orientation\n"
                                       "  l = save ptofile\n"
                                       "  s = save event.txt\n"
                                       "  S = save centroid.txt\n"
                                       "  left arrow = back one image\n"
                                       "  right arrow = forward one image\n"
                                       "  page up = back ten images\n"
                                       "  page down = forward ten images\n"
                                       "  mouse wheel = zoom\n"
                                       "  left mouse button = drag image\n"
                                       "  right mouse button = mark position and forward",
                                       fill="yellow", font=("helvetica", 10))
                self.overlay.append(text_id)

            for s in stars:
                x, y = (s[0] - self.offsetx) * self.imscale + bbox[0], (s[1] - self.offsety) * self.imscale + bbox[1]
                self.overlay.append(self.canvas.create_oval(x, y, x, y, width=(max(0.1, 5-s[5])*2 if self.show_info else 1), outline="red"))
                if self.show_info == 1:
                    self.overlay.append(self.canvas.create_text(x+3, y+3, text=s[4] + " (" + str(s[5]) + ")", anchor="nw", fill="green", font=("helvetica", 10)))

                #self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection
            
                
    def click(self, event):
        x, y = event.x / self.imscale + self.x, event.y / self.imscale + self.y
        x += self.offsetx
        y += self.offsety
        positions.append((x, y))
        inv.transformImgCoord(dst, hsi.FDiff2D(x, y))
        #print('{} {}'.format(round(x, 2), round(y, 2)))
        if self.timestamps:
            diff = self.timestamps[self.num] - self.timestamps[0]
            ts = datetime.fromtimestamp(self.timestamps[self.num], UTC)
        else:
            diff = self.num / args.fps
            ts = starttime + timedelta(seconds=round(self.num / args.fps, 2))
        centroid[self.num] = '{} {} {} {} 1.0 {} {}'.format(self.num,
                                                            round(diff, 2),
                                                            round(90 - (dst.y / 100), 2), round((dst.x / 100) % 360, 2),
                                                            args.name, ts.strftime("%Y-%m-%d %H:%M:%S.%f UTC"))
        print(centroid[self.num])
        if self.num < len(self.files) - 1:
            self.num += 1
            self.image = Image.open(self.files[self.num])
            self.show_image()
        else:
            exit(0)



temp = []
images = []

for i in args.imgfiles:
    if i.endswith('.mp4'):
        temp.append(tempfile.TemporaryDirectory(prefix="clickcoords_", dir="/tmp"))
        try:
            read_frames(i, temp[-1].name)
            for file in sorted(glob.glob(temp[-1].name + "/*.tif")):
                images.append(file)
        except Exception as e:
            print(e)
            pass
    else:
        images.append(os.getcwd() + "/" + i)

timestamps = None
positions = []

if args.start is None:
    # First, extract all timestamps and populate the cache.
    raw_timestamps = [None] * len(images)
    total_images = len(images)
    bar_length = 40
    
    print("Extracting timestamps...")
    if total_images > 0:
        for i in range(total_images):
            timestamp(raw_timestamps, images, i) # This populates the list
            
            # Update progress bar
            progress = (i + 1) / total_images
            block = int(round(bar_length * progress))
            text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {i+1}/{total_images} ({progress*100:.0f}%)"
            sys.stdout.write(text)
            sys.stdout.flush()
        sys.stdout.write("\n\n")

    # Second, run the interpolation on the fully populated list of timestamps.
    print("Interpolating timestamps...")
    timestamps = interpolate_timestamps(raw_timestamps)
    if timestamps:
        starttime = timestamps[0]
    else:
        starttime = datetime.now().timestamp() # Fallback
else:
    starttime = datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S.%f').timestamp()
    timestamps = [None] * len(images)
    for x in range(len(images)):
        timestamps[x] = starttime + (x / args.fps)

    
centroid = [None] * len(images)

first = Image.open(images[0])
pano = hsi.Panorama()
try:
    pano.ReadPTOFile(args.ptofile)
except:
    pano.readData(hsi.ifstream(args.ptofile))
img = pano.getImage(args.image)
pitch = img.getPitch()
yaw = img.getYaw()
roll = img.getRoll()
zoom = img.getHFOV()
radial = img.getRadialDistortion()
radial_shift = img.getRadialDistortionCenterShift()
img.setSize(hsi.Size2D(first.width, first.height))
width = img.getSize().width()
height = img.getSize().height()
scale = int(pano.getOptions().getWidth() / pano.getOptions().getHFOV())
dst = hsi.FDiff2D()
inv = hsi.Transform()
inv.createInvTransform(img, pano.getOptions())

window = tk.Tk()
window.geometry("1600x960")
app = Zoom_Advanced(window, files=images, timestamps=timestamps)
window.mainloop()

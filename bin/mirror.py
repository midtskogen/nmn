#!/usr/bin/env python3

import inotify.adapters
import argparse
import logging
import re
import shlex
import subprocess
import os
import threading
import asyncio
import time
import datetime
import tempfile
import shutil
import stat
from collections import deque

parser = argparse.ArgumentParser(description='Monitor AMS directories for new files.')
parser.add_argument('-l', '--logfile', dest='logfile', help='logfile')
parser.add_argument('-d', '--delay', dest='videodelay', help='video delay in ms (default=150)', default=150)
parser.add_argument('-L', '--link', dest='link', help='create link instead of copy', action="store_true")
parser.add_argument('-r', '--remotedir', dest='remotedir', help='destination directory', default='/meteor')
parser.add_argument('-e', '--execute', dest='exefile', help='program to pass event file to (default: "/home/meteor/bin/ams2event.py")', default="/home/meteor/bin/ams2event.py")
parser.add_argument(action='store', dest='recdir', help='directory to be monitored recursively (for detections)')
parser.add_argument(nargs='*', action='store', dest='dirs', help='additional directories to be monitored non-recursively (for video files)')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.logfile, level=logging.INFO)

processed_files = deque(maxlen=100)

async def watch_videos():
    logging.info('Enter watch_videos')
    while True:
        i = inotify.adapters.Inotify()
        events = i.event_gen(yield_nones=False)
        for d in args.dirs:
            i.add_watch(d, mask=inotify.constants.IN_CLOSE_WRITE + inotify.constants.IN_MOVED_TO)

        for event in events:
            try:
                (_, _, eventdir, eventfile) = event
                if eventdir + eventfile in processed_files:
                    #logging.info('Skipping: ' + eventdir + '/' + eventfile)
                    continue
                #logging.info('Event: ' + eventdir + '/' + eventfile)
                try:
                    (year, month, day, hour, minute, second, zero, id, ext) = re.split('_|\\.', eventfile)
                    dt = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

                    # Get the file's birth time (btime) or access time (atime) with fractional seconds
                    stat = os.stat(os.path.join(eventdir, eventfile))
                    try:
                        file_time = datetime.datetime.fromtimestamp(stat.st_birthtime)  # Try to get birth time
                    except AttributeError:
                        file_time = datetime.datetime.fromtimestamp(stat.st_atime)  # Fall back to access time
                    
                    # Check if file_time is within 2 seconds of dt
                    if abs((file_time - dt).total_seconds()) <= 2:
                        # There seems to be a delay of about 150 milliseconds from real event to file creation
                        dt = file_time - datetime.timedelta(milliseconds=args.videodelay)
                    
                    offset = dt.timestamp()
                    # Add 20 seconds, since somethimes the timestamp is XX:YY:00, sometimes XX:YY:59 to avoid double full/mini_YY.mp4
                    dt = dt + datetime.timedelta(seconds=20)
                    (year, month, day, hour, minute, second) = (str(dt.year), str(dt.month).zfill(2), str(dt.day).zfill(2), str(dt.hour).zfill(2), str(dt.minute).zfill(2), str(dt.second).zfill(2))
                    if id.isnumeric():
                        cam = int(id) % 10
                    else:
                        continue
                except:
                    logging.exception('Got exception!')
                    continue
                    
                if ext != 'mp4' or os.stat(eventdir + '/' + eventfile).st_size == 0:
                    continue



                # Add the timestamp into the file via a temporary file
                original_file = os.path.join(eventdir, eventfile)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    modified_file = temp_file.name
                try:
                    ffmpeg_cmd = f"ffmpeg -y -i {shlex.quote(original_file)} -c copy -movflags +faststart -metadata creation_time={datetime.datetime.fromtimestamp(offset).isoformat()}Z -output_ts_offset {offset} {shlex.quote(modified_file)}"
                    try:
                        #logging.info(f"Running {ffmpeg_cmd}")
                        subprocess.check_output(shlex.split(ffmpeg_cmd), stderr=subprocess.PIPE, timeout=30)
                        copy_cmd = ['cp', '-f', modified_file, original_file]
                        subprocess.check_output(copy_cmd, stderr=subprocess.STDOUT)
                        #logging.info(f"Changed timestamp on {shlex.quote(original_file)} to {offset} - {datetime.datetime.fromtimestamp(offset).isoformat()}")                                                     
                    except subprocess.CalledProcessError as e:
                        logging.exception(f"Got exception! Command output: {e.output.decode()}")
                except:
                    logging.exception('Got exception!')
                    
                # Remove the temporary file after successful copy
                os.remove(modified_file)

                processed_files.append(eventdir + eventfile)

                remotedir = args.remotedir + '/cam' + str(cam) + '/' + year + month + day + '/' + hour
                remotefile = ('mini_' if eventdir.endswith('SD') else 'full_') + minute + '.mp4'
                jpgfile = ('mini_' if eventdir.endswith('SD') else 'full_') + minute + '.jpg'
    
                os.makedirs(remotedir, exist_ok=True)

                if eventdir.endswith('HD'):
                    cmd = '/home/meteor/bin/vid2jpg.sh ' + eventdir + '/' + eventfile + ' ' + remotedir + '/' + jpgfile
                    #logging.info('Executing: ' + cmd)
                    subprocess.check_output(shlex.split(cmd), timeout=30)
                    if args.link:
                        cmd = 'ln -f ' + remotedir + '/' + jpgfile + ' ' + args.remotedir + '/cam' + str(cam) + '/snapshot.jpg'
                        #logging.info('Executing: ' + cmd) 
                        subprocess.check_output(shlex.split(cmd), stderr=subprocess.PIPE, timeout=30)

                if args.link:
                    cmd = 'ln ' + eventdir + '/' + eventfile + ' ' + remotedir + '/' + remotefile
                else:
                    cmd = 'cp ' + eventdir + '/' + eventfile + ' ' + remotedir + '/' + remotefile
                #logging.info('Executing: ' + cmd)
                process = subprocess.check_output(shlex.split(cmd), stderr=subprocess.PIPE, timeout=30)
                #for line in process.stderr:
                    #logging.info('%r', line)

                #logging.info('Executed: ' + cmd)
            except:
                logging.exception('Got exception!')
                continue
        logging.info('Loop watch_videos')
    logging.info('Exit watch_videos')

async def watch_detections():
    logging.info('Enter watch_detections')
    while True:
        detections = inotify.adapters.InotifyTree(args.recdir, inotify.constants.IN_CLOSE_WRITE).event_gen(yield_nones=False)
        for event in detections:
            try:
                (_, type, eventdir, eventfile) = event
                file = eventfile.split("_")
                dir = eventdir.split('/')
                if 'IN_CLOSE_WRITE' in type and dir[-2] == 'meteors' and file[-1].split('-')[-1] == 'reduced.json':
                    logging.info('Executing: ' + args.exefile + ' ' + eventdir + '/' + eventfile)
                    proc = subprocess.check_output([args.exefile, eventdir + '/' + eventfile], timeout=30)
            except:
                logging.exception('Got exception!')
                continue
        logging.info('Loop watch_detections')
    logging.info('Exit watch_detections')

def main():
    t1 = threading.Thread(target=asyncio.run, args=(watch_detections(), ))
    t2 = threading.Thread(target=asyncio.run, args=(watch_videos(), ))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

if __name__ == '__main__':
    main()

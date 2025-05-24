#!/usr/bin/env python3

# Usage: cammon.py [-l <logfile>] [-d <format>] [-x <maxfile>] [-b <minutes> ] [ -s <statefile> ] <dir>
# This program monitors directories for new files given by <format>.
# Raw H.264 will be extracted from these and output to stdout for reading by metdetect.

import inotify.adapters
import datetime
import argparse
import os
import pathlib
import time
import glob
import logging
import errno
import signal
import shutil

from ffmpy import FFmpeg

parser = argparse.ArgumentParser(description='Monitor a directory for new files and extract H.264.')
parser.add_argument('-l', '--logfile', dest='logfile', help='logfile')
parser.add_argument('-d', '--format', dest='format', help='file format (default: %Y%m%d/%H/full_%M)', default='%Y%m%d/%H/full_%M')
parser.add_argument('-x', '--maxfile', dest='maxfile', help='file which contains the location of an hourly max file used by metdetect')
parser.add_argument('-b', '--backlog', dest='backlog', help='number of minutes of backlog to look for', default=1)
parser.add_argument('-s', '--statefile', dest='statefile', help='save the most recently processed file to this file')
parser.add_argument(action='store', dest='dir', help='directory to be monitored recursively')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.logfile, level=logging.INFO)

class TimeoutError(Exception):
#     logging.warn('Error processing ' + cur + ": " + e)
#     sys.exit(-1)
    pass

class timeout:
    def __init__(self, seconds=60, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

if not args.dir.endswith('/'):
    args.dir = args.dir + '/'

args.backlog = max(1, int(args.backlog))
i = inotify.adapters.Inotify()
events = i.event_gen()

def add_watches(dirs):
    for dir in dirs:
        logging.info('Add watch: ' + args.dir + dir)
        pathlib.Path(args.dir + dir).mkdir(parents=True, exist_ok=True)
        i.add_watch(args.dir + dir, mask=8+128)

def remove_watches(dirs):
    for dir in dirs:
        logging.info('Remove watch: ' + args.dir + dir)
        i.remove_watch(args.dir + dir)

# Parse timestamp in file and ignore unmatched remainder
def get_time(file):
    return time.strptime(file[:len(datetime.datetime.now().strftime(args.dir + args.format))], args.dir + args.format)
    
def rewind(eventdir, minutes):
    result = []
    for m in reversed(range(0, max(1, minutes))):
        try:
            t = datetime.datetime.fromtimestamp(time.mktime(get_time(eventfile))) - datetime.timedelta(minutes=m)
            result.extend(glob.glob(args.dir + t.strftime(args.format) + '*'))
        except:
            pass
    return result

watches = prev_watches = set()
thishour = nexthour = prevhour = lasteventfile = ''
tophour = datetime.datetime.fromtimestamp(0, datetime.UTC)

import sys
for event in events:
    now = datetime.datetime.now();
    if args.maxfile != None and os.path.dirname(now.strftime(args.format)) != os.path.dirname(tophour.strftime(args.format)):
        tophour = now;
        try:
            # Update the file atomically
            f = open(args.maxfile + '.tmp', 'w')
            f.write(args.dir + os.path.dirname(now.strftime(args.format)) + '/max.jpg\n')
            f.flush()
            f.close()
            os.rename(args.maxfile + '.tmp', args.maxfile)
        except:
            logging.warn('Could not write maxfile')
        
    now = now + datetime.timedelta(minutes=30)
    if os.path.dirname(now.strftime(args.format)) != thishour:
        thishour = os.path.dirname(now.strftime(args.format))
        nexthour = os.path.dirname((now + datetime.timedelta(hours=1)).strftime(args.format))
        prevhour = os.path.dirname((now - datetime.timedelta(hours=1)).strftime(args.format))
        watches = { thishour, nexthour, prevhour }
        remove_watches(prev_watches - watches)
        add_watches(watches - prev_watches)
        prev_watches = watches

    if event == None:
        continue
    (_, _, eventdir, eventfile) = event
    eventfile = eventdir + '/' + eventfile
    file = args.dir + args.format
    try:
        t = get_time(eventfile)
    except:
        t = None

    if t == None or eventfile == lasteventfile:
        continue

    prev = eventfile
    try:
        with open(args.statefile, 'r') as f:
            prev = f.read().strip()
    except:
        if args.statefile != None:
            logging.warn('Could not read statefile')

    lasteventfile = eventfile
    try:
        prevtime = get_time(prev)
    except:
        prevtime = t
    
    files = rewind(eventdir, min(args.backlog, int((time.mktime(t) - time.mktime(prevtime)) / 60)))
    logging.info(files)
    
    for cur in files:
        #sd = cur.replace('full', 'mini')
        jpg_hd = prev.replace('mini', 'full').replace('mp4', 'jpg')
        try:
            shutil.copyfile(jpg_hd, jpg_hd + '.tmp')
            os.rename(jpg_hd + '.tmp', args.dir + '/snapshot.jpg')
        except:
            pass
        #if sd != cur and os.path.isfile(sd):
        #    cur = sd
        ff = FFmpeg(inputs={cur: None}, outputs={'pipe:1': '-nostdin -vsync 0 -timeout 3 -loglevel quiet -f rawvideo -c:v copy -c:a none -bsf:v h264_mp4toannexb'})
        logging.info('Running: ' + ff.cmd)
        try:
            with timeout(seconds=120):
                ff.run()

            logging.info('Finished processing ' + cur + ': ' + str(ff.process.communicate()))
            ff.process.kill()
            if args.statefile != None:
                try:
                    f = open(args.statefile, 'w')
                    f.write(cur + '\n')
                    f.close()
                except:
                    logging.warn('Could not write statefile')

        except Exception as e:
            logging.warning('Error processing ' + cur + ": " + str(e))

#!/usr/bin/env python3

import inotify.adapters
import argparse
import os
import pathlib
from dvrip import DVRIPCam

parser = argparse.ArgumentParser(description='Monitor AMS video directories.')
parser.add_argument(action='store', dest='dir', help='directory to be monitored')
parser.add_argument(action='store', dest='symdir', help='directory for symlinks')
args = parser.parse_args()

i = inotify.adapters.Inotify()
events = i.event_gen()

i.add_watch(args.dir + "/HD", mask=8+128)
i.add_watch(args.dir + "/SD", mask=8+128)

for event in events:
    if event == None:
        continue

    (_, _, eventdir, eventfile) = event
    file = eventfile.split("_")
    try:
        year = file[0]
        month = file[1]
        day = file[2]
        hour = file[3]
        minute = file[4]
        second = file[5]
        id = os.path.splitext(file[7])[0]
        camid = id[-1]
        quality = eventdir[-2] + eventdir[-1]
        eventfile = eventdir + '/' + eventfile
        nmnfile = args.symdir + "/cam" + camid + "/" + year + month + day + "/" + hour + ("/full_" if quality == "HD" else "/mini_") + minute + ".mp4"
        pathlib.Path(os.path.dirname(nmnfile)).mkdir(parents=True, exist_ok=True)
        if os.path.exists(nmnfile):
            count = 1
            while True:
                nmnfile2 = os.path.splitext(nmnfile)[0] + "_" + str(count) + ".mp4"
                count = count + 1
                if not os.path.exists(nmnfile2) or count > 9:
                    break
            nmnfile = nmnfile2
        os.symlink(eventfile, "/tmp/amsmon_link")
        os.rename("/tmp/amsmon_link", nmnfile)
        
        ip = "192.168.76.7" + camid
        if minute == "00" and quality == "HD":
            print(ip)
            cam = DVRIPCam(ip, user='admin', password='')
            if cam.login():
                cam.set_time()
                cam.close()

    except Exception as e:
        print(e)
        pass

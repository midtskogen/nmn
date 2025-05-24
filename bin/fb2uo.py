#!/usr/bin/python3

class AlertInfo:
    '''
    Structure for information on a single event
    '''
    def __init__(self):
        self.starttime = 0
        self.filename = ''
        self.age = 0
        self.datline = ''
        self.site = ''
        self.datestr = ''
        self.timestr = ''


def findalerts(lookfor):

    import glob

    dirs = glob.glob(lookfor)
    dirs.sort()

    alertlist = []
    for d in dirs:
        alertname = glob.glob(d + '/alert_*.txt')
        if len(alertname) > 0:
            alertlist.append(alertname[0])

    return alertlist



def get_alert_info(filename):
    '''
    Extract data from an alert-file and return them in a structure
    '''
    import time

    f = open(filename, 'r')
    data = f.readlines()
    info = AlertInfo()
    # Adjust line offset depending on whether directory name has been prepended 
    offset = 0
    if data[0][:2] == 'An':
        offset = -1
    # Also extracts timezone, if present, but not currently used
    [site, datestr, timestr] = data[3+offset].split(None, 2)
    try: [timestr, timezone] = timestr.split()
    except ValueError: timestr = timestr.strip()

    info.site = site
    info.datestr = datestr
    info.timestr = timestr[:-3]
    starttime = time.strptime(datestr+' '+timestr[:-3], \
    '%Y-%m-%d %H:%M:%S')
    fstarttime = time.mktime(starttime) + float(timestr[-3:])

    print(datestr + '   ' + timestr)

    info.starttime = fstarttime
    info.filename = filename
    info.datline = data[6+offset]
    if info.datline[-1]=='\n':
        info.datline=info.datline[:-1]
    return info



def is_number(s):
    # also in metrack.py
    try:
        float(s)
        return True
    except ValueError:
        return False



def parse_datline(line):
    words=line.split()
    lon=words[0]
    lat=words[1]
    az1=words[2]
    az2=words[3]
    alt1=words[4]
    alt2=words[5]
    duration=words[7]
    nam=''
    nam_nsegs=0
    for n in words[12:]:
        if not is_number(n):
            nam=nam+' '+n
            nam_nsegs += 1
    nam=nam[1:]
    # Unix timestamp not used
    #try:
    #    unixsec=float(words[12+nam_nsegs])
    #except:
    #    unixsec = 0.
    try:
        obsheight=int(words[13+nam_nsegs])
    except:
        obsheight = 0
        heightdict = { 'CPH' : 30, \
                       'KLH' : 45, \
                       'AAL' : 52, \
                       'HOB' : 62, \
                       'SIL' : 90, \
                       'ORI' : 48 } 
        try:
            obsheight = heightdict[nam]
        except:
            obsheight=0

    return lon, lat, az1, az2, alt1, alt2, duration, nam, obsheight



def fb2uo(lookfor, output):
    import time
    #import ephem
    #from numpy import degrees as deg

    # Scan folders with Fireball Network scructure
    # and write contents to CSV file as defined by UFOOrbit format
    # http://sonotaco.com/soft/UO2/UO21Manual_EN.pdf

    ver = 'R91'
    mag = '0' # Use 0 for unknown
    al = findalerts(lookfor)
    f = open(output,'w')
    f.write('Ver,Y,M,D,h,m,s,Mag,Dur,Az1,Alt1,Az2,Alt2, Ra1, Dec1, Ra2, Dec2,ID,Long,Lat,Alt,Tz\n')
    for an in al:
        info = get_alert_info(an)
        lon, lat, az1, az2, alt1, alt2, duration, nam, obsheight = parse_datline(info.datline)

        namedict = { 'CPH' : 'Copenhagen', \
                     'KLH' : 'Klokkerholm', \
                     'AAL' : 'Aalborg', \
                     'HOB' : 'Hobro', \
                     'SIL' : 'Silkeborg', \
                     'ORI' : 'Jels' }
        try:
            name = namedict[nam]
        except:
            name = nam

        #t = time.gmtime(info.starttime) # UT
        #tz = 0 # UT
        # Local time and timezone required by UFOOrbit
        t = time.localtime(info.starttime) 
        tz = -time.timezone/3600 + t.tm_isdst

        # Unused: Alt, Az are provided, so do not calculate RA, Dec
        #obs = ephem.Observer()
        #obs.lon = lon 
        #obs.lat = lat
        #obs.elevation = obsheight
        #ts = time.strftime('%Y/%m/%d %H:%M:%S' ,t)
        #obs.date = ephem.Date(ts)
        #ra1,dec1 = obs.radec_of(az1, alt1)
        #ra2,dec2 = obs.radec_of(az2, alt2)
        #ra1 = deg(ra1)
        #ra2 = deg(ra2)
        #dec1 = deg(dec1)
        #dec2 = deg(dec2)
        ra1 = 999.9
        dec1 = 999.9
        ra2 = 999.9
        dec2 = 999.9

        # Convert Az to UFOOrbit convention
        uaz1 = '%.1f' % ((float(az1)+180) % 360)
        uaz2 = '%.1f' % ((float(az2)+180) % 360)

        csvline = '%s,%i,%i,%i,%i,%i,%.2f,%s,%s,%s,%s,%s,%s,%.1f,%.1f,%.1f,%.1f,%s,%s,%s,%i,%i' \
        % (ver, t.tm_year,t.tm_mon,t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec + (info.starttime % 1), mag, duration, uaz1, alt1, uaz2, alt2, ra1, dec1, ra2, dec2, name, lon, lat, obsheight, tz)
        #print(csvline)
        f.write('%s\n' % csvline)

    f.close()


if __name__ == "__main__":
    import optparse
    parser = optparse.OptionParser(description='',
        version='',
        epilog='' )
    parser.add_option('-l', '--lookfor', dest='lookfor', default='event*',
        help='Directory pattern to search for alert files in. E.g. "event2014"')

    parser.add_option('-o', '--output', dest='output', default='touo.csv',
        help='Name for output CSV file.')

    options, remainder = parser.parse_args()

    if options.lookfor == 'event*' and len(remainder):
        options.lookfor = remainder[0]

    fb2uo(options.lookfor, options.output)

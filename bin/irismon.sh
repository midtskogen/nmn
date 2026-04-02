#!/bin/bash                                                                                                                                          

sunalt=$(printf "%.0f\n" $(/home/meteor/bin/sunpos.py | cut -d" " -f2))

if [[ $(expr $sunalt+0) -gt -4 ]]; then
    if [[ $(cat /tmp/iris.txt 2> /dev/null) != "day" ]] || [[ $(find /tmp/iris.txt -mmin +15 2> /dev/null) ]]; then
        /home/meteor/bin/iris_day.sh
        if [[ $? -eq 0 ]]; then echo day > /tmp/iris.txt; fi
    fi
else
    if [[ $(cat /tmp/iris.txt 2> /dev/null) != "night" ]] || [[ $(find /tmp/iris.txt -mmin +15 2> /dev/null) ]]; then
        /home/meteor/bin/iris_night.sh
        if [[ $? -eq 0 ]]; then echo night > /tmp/iris.txt; fi
    fi
fi

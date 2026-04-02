#!/bin/bash

ip=$(ifconfig | grep "inet " | awk '{print $2}' | sed 's/addr://' | grep -v 127.0.0 | head -n1)
hosts=$(cat /etc/cameras.ip)
ids=$(cat /etc/cameras.id)
arr=($ids)

base64 -d >/tmp/custom_logo.jpg << ENDOFFILE
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
JUlHNq8di0sPKtCEA3/dETBNe32++5N7EQGgXNpL82r7HbKeLHSTxitYOmd/JjR8cx0HEOb22hjT
6qnonroo2GYKGF2P6G9VeV0sz1LKllMteaAm0HxgDken7Yn4xN9GITp93C8dVlZdnWYdYO461T7x
WEPMBaL0A9sQ2S4qv1DPn3NUW/CzRPUHWFOTNZf3jpz74zG84eskjsuceGjJur5UC81q+UdRdouV
11VkQLw+SObk6APeGoPwFMHdvHehHWVnHF/itBMs09vbD9qjNdGpKYctMD11Zdpn8m5UXxJ4RZQn
1MtR+JvOQgKFrJnu+TL8RmtfqyZX13J9+V0HrI/tw/Tv9jE103UXhSZSgpnmSpRo2AEsSi7g3OUx
vWlCJVp6uMuN4dO10JYGdm/ZnNxVf0nx+cNmsBqruxwnyxvA1bPBhAeG+FrEPiMB5c3Bw8p7u+85
jnxXfofkXONK4hDTQH/81DPDgKrlNapUJCalyMoxb5aKaUfkPwCKaQPO8VuafAKnD9eP34uzXZw8
9uGSyK1OpMagfGMHjLfxOl/SdHZdVPopfvz/siBGnyLYsxtgjot+jmg8xssxhJGfXoyvZshycQzQ
GqpHK5Tj3/O7do9Er8oHQR5jcOi/xztNPsgL4uyMsnvzuf4+NjaiejyaDaBGfsT7TplQ0pZohCC1
dDYgHZc1T1vhxdoXAehtoDe5DUJwn8HJ/Q0kJtuSWBHW483lUGngMalV2kpmjORidF4DjclkO4DE
OjQUh0288eajDQgnbv8He2w7D8wwtC8AAAAASUVORK5CYIINCi0tLS0tLS0tLS0tLS0tLS0tLS0t
LS0tLS0tLS0tNjI3MzA1MzMwMzk1NjY5NjU3Mzc3MDAxMzcNCkNvbnRlbnQtRGlzcG9zaXRpb246
IGZvcm0tZGF0YTsgbmFtZT0ibGF5b3V0X2N1c3RvbWJ1dHRvbl9tYW51YWx0cmlnZ2VyX3Nob3ci
DQoNCjENCi0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tMTQ1MjAyNzM2MTM1MDU0MjI5NTE3
MzA2ODk2MDUNCkNvbnRlbnQtRGlzcG9zaXRpb246IGZvcm0tZGF0YTsgbmFtZT0ibGF5b3V0X2N1
c3RvbWJ1dHRvbl9tYW51YWx0cmlnZ2VyX3Nob3ciDQoNCjE=
ENDOFFILE

 
for i in $hosts; do
    let c=c+1
    n=${arr[c-1]}
    if ! ping -c 1 $i &> /dev/null; then continue; fi

    cam=$(curl -su root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/getparam.cgi?system_info_modelname" | awk -F\' '{ print $2 }')

    echo Configuring $i \($n, $cam\):
    curl -su root:XXCAMPASSWORDXX "http://$i/cgi-bin/admin/setparam.cgi?network_ftp_enable=1"
    echo "ftp     stream  tcp     nowait  root    /usr/sbin/ftpd ftpd" > /tmp/inetd.conf
    echo "telnet  stream  tcp     nowait  root    /usr/sbin/telnetd telnetd -i" >> /tmp/inetd.conf
    echo "cd /etc; put -f /tmp/inetd.conf" | ncftp -u root -p XXCAMPASSWORDXX $i
    rm -f /tmp/inetd.conf
    echo "mkdir /mnt/flash/www; mkdir /mnt/flash/www/pic" | ncftp -u root -p XXCAMPASSWORDXX $i
    ncftpput -u root -p XXCAMPASSWORDXX $i /mnt/flash/www/pic /tmp/custom_logo.jpg

    if [[ "$cam" == "IP8151" ]]; then
	for j in \
	    system_hostname=XXSTATIONXX-$n \
 	    system_ntp=$ip \
	    system_timezoneindex=0 \
 	    system_daylight_auto_begintime=Disabled \
	    system_daylight_auto_endtime=Disabled \
	    system_updateinterval=3600 \
	    seamlessrecording_enable=1 \
	    network_rtsp_s0_audiotrack=-1 \
	    network_rtsp_s1_audiotrack=-1 \
	    network_rtsp_s2_audiotrack=-1 \
	    network_qos_dscp_video=63 \
	    videoin_cmosfreq=60 \
	    videoin_whitebalance=rbgain \
	    videoin_imprinttimestamp=1 \
	    videoin_maxexposure=5 \
	    videoin_c0_textonvideo_size=15 \
	    videoin_c0_textonvideo_position=bottom \
	    videoin_c0_cmosfreq=60 \
	    videoin_c0_whitebalance=rbgain \
	    videoin_c0_rgain=56 \
	    videoin_c0_bgain=20 \
	    videoin_c0_wdrc_mode=1 \
	    videoin_c0_maxgain=30 \
	    videoin_c0_imprinttimestamp=1 \
	    videoin_c0_maxexposure=5 \
	    videoin_c0_profile_i0_maxexposure=5 \
	    videoin_c0_s0_codectype=h264 \
	    videoin_c0_s0_resolution=1280x1024 \
	    videoin_c0_s0_mpeg4_intraperiod=250 \
	    videoin_c0_s0_h264_intraperiod=4000 \
	    videoin_c0_s0_h264_ratecontrolmode=vbr \
	    videoin_c0_s0_h264_quant=99 \
	    videoin_c0_s0_h264_qvalue=26 \
	    videoin_c0_s0_h264_qpercent=88 \
	    videoin_c0_s0_h264_bitrate=4000000 \
	    videoin_c0_s0_h264_maxframe=30 \
	    videoin_c0_s0_h264_profile=2 \
	    videoin_c0_s0_h264_maxvbrbitrate=4000000 \
	    videoin_c0_s0_h264_prioritypolicy=framerate \
	    videoin_c0_s1_codectype=h264 \
	    videoin_c0_s1_resolution=800x600 \
	    videoin_c0_s1_codectype=h264 \
	    videoin_c0_s1_h264_maxframe=30 \
	    videoin_c0_s1_mpeg4_ratecontrolmode=vbr \
	    videoin_c0_s1_h264_intraperiod=4000 \
	    videoin_c0_s1_h264_quant=99 \
	    videoin_c0_s1_h264_qvalue=26 \
	    videoin_c0_s1_h264_bitrate=2000000 \
	    videoin_c0_s1_h264_profile=2 \
	    videoin_c0_s1_h264_maxvbrbitrate=2000000 \
	    videoin_c0_s1_h264_prioritypolicy=framerate \
	    videoin_c0_s2_h264_quant=99 \
	    videoin_c0_s2_h264_qvalue=24 \
	    videoin_c0_s2_h264_maxframe=1 \
	    videoin_c0_s2_h264_maxvbrbitrate=10000000 \
	    videoin_c0_s2_mjpeg_ratecontrolmode=cbr \
	    videoin_c0_s2_mjpeg_bitrate=10000000 \
	    videoinpreview_maxexposure=5 \
	    videoinpreview_wdrc_mode=1 \
	    videoinpreview_maxgain=30 \
	    image_c0_saturation=100 \
	    image_c0_saturationpercent=59 \
	    image_c0_sharpness=100 \
	    image_c0_sharpnesspercent=0 \
	    image_c0_lowlightmode=0 \
	    imagepreview_c0_saturation=100 \
	    imagepreview_c0_saturationpercent=59 \
	    imagepreview_c0_sharpness=100 \
	    imagepreview_c0_sharpnesspercent=0 \
	    imagepreview_c0_lowlightmode=0 \
	    imagepreview_videoin_whitebalance=rbgain \
	    imagepreview_videoin_rgain=64 \
	    imagepreview_videoin_bgain=16 \
	    motion_c0_win_i0_enable=1 \
	    motion_c0_win_i0_name=ildkule \
	    motion_c0_win_i0_width=320 \
	    motion_c0_win_i0_height=171 \
	    motion_c0_win_i0_objsize=10 \
	    motion_c0_win_i0_sensitivity=90 \
	    audioin_c0_mute=1 \
	    server_i0_name=odroid \
	    server_i0_type=ns \
	    server_i0_ns_location=\\\\$ip\\meteor$n \
	    server_i0_ns_username=meteor \
	    server_i0_ns_passwd=XXPASSWORDXX \
	    recording_i0_name=1280x1024 \
	    recording_i0_enable=1 \
	    recording_i0_source=0 \
	    recording_i0_limitsize=0 \
	    recording_i0_cyclic=1 \
	    recording_i0_prefix=full_ \
	    recording_i0_reserveamount=16384 \
	    recording_i0_dest=0 \
	    recording_i0_adaptive_preevent=5 \
	    recording_i0_adaptive_postevent=5 \
	    recording_i1_name=800x600 \
	    recording_i1_enable=1 \
	    recording_i1_source=1 \
	    recording_i1_limitsize=0 \
	    recording_i1_cyclic=1 \
	    recording_i1_prefix=mini_ \
	    recording_i1_reserveamount=4194304 \
	    recording_i1_dest=0 \
	    recording_i1_adaptive_preevent=5 \
	    recording_i1_adaptive_postevent=5 \
	    ircutcontrol_mode=day \
	    ircutcontrol_bwmode=0 \
	    ircutcontrol_sensitivity=low \
	    layout_logo_default=0 \
	    layout_logo_link=http://norskmeteornettverk.no \
	    layout_logo_powerbyvvtk_hidden=1 \
	    disk_i0_cyclic_enabled=1 \
	    roi_c0_s0_home=0,0 \
	    roi_c0_s0_size=1280x1024 \
	    roi_c0_s1_home=0,0 \
	    roi_c0_s1_size=1280x1024 \
	    privacymask_c0_win_i0_enable=1 \
	    privacymask_c0_win_i0_name=clock \
	    privacymask_c0_win_i0_left=0 \
	    privacymask_c0_win_i0_top=237 \
	    privacymask_c0_win_i0_width=22 \
	    privacymask_c0_win_i0_height=3 \
            privacymask_c0_enable=1 \
            network_ftp_enable=1 \
	    ;
	do curl -u root:XXCAMPASSWORDXX "http://$i/cgi-bin/admin/setparam.cgi?$j"; sleep 0.2
	done
    fi

    if [[ "$cam" == "IP816A-HP-LPC(Street)" ]]; then
	for j in \
	    system_hostname=XXSTATIONXX-$n \
 	    system_ntp=$ip \
	    system_timezoneindex=0 \
 	    system_daylight_auto_begintime=Disabled \
	    system_daylight_auto_endtime=Disabled \
	    system_updateinterval=3600 \
	    seamlessrecording_enable=1 \
	    network_rtsp_s0_audiotrack=-1 \
	    network_rtsp_s1_audiotrack=-1 \
	    network_rtsp_s2_audiotrack=-1 \
	    network_qos_dscp_video=63 \
	    videoin_cmosfreq=60 \
	    videoin_whitebalance=rbgain \
	    videoin_imprinttimestamp=1 \
	    videoin_maxexposure=5 \
	    videoin_c0_textonvideo_size=15 \
	    videoin_c0_textonvideo_position=bottom \
	    videoin_c0_cmosfreq=50 \
	    videoin_c0_whitebalance=rbgain \
	    videoin_c0_rgain=75 \
	    videoin_c0_bgain=40 \
	    videoin_c0_wdrc_mode=1 \
	    videoin_c0_maxgain=40 \
	    videoin_c0_imprinttimestamp=1 \
	    videoin_c0_maxexposure=5 \
	    videoin_c0_profile_i0_maxexposure=5 \
	    videoin_c0_s0_codectype=h264 \
	    videoin_c0_s0_resolution=1920x1080 \
	    videoin_c0_s0_mpeg4_intraperiod=250 \
	    videoin_c0_s0_h264_intraperiod=4000 \
	    videoin_c0_s0_h264_ratecontrolmode=vbr \
	    videoin_c0_s0_h264_quant=99 \
	    videoin_c0_s0_h264_qvalue=26 \
	    videoin_c0_s0_h264_qpercent=88 \
	    videoin_c0_s0_h264_bitrate=4000000 \
	    videoin_c0_s0_h264_maxframe=5 \
	    videoin_c0_s0_h264_profile=2 \
	    videoin_c0_s0_h264_maxvbrbitrate=40000000 \
	    videoin_c0_s0_h264_prioritypolicy=framerate \
	    videoin_c0_s1_codectype=h264 \
	    videoin_c0_s1_h264_maxframe=5 \
	    videoin_c0_s1_resolution=800x600 \
	    videoin_c0_s1_mpeg4_ratecontrolmode=vbr \
	    videoin_c0_s1_h264_intraperiod=4000 \
	    videoin_c0_s1_h264_quant=99 \
	    videoin_c0_s1_h264_qvalue=26 \
	    videoin_c0_s1_h264_bitrate=2000000 \
	    videoin_c0_s1_h264_profile=2 \
	    videoin_c0_s1_h264_maxvbrbitrate=20000000 \
	    videoin_c0_s1_h264_prioritypolicy=framerate \
	    videoin_c0_s2_h264_quant=99 \
	    videoin_c0_s2_h264_qvalue=24 \
	    videoin_c0_s2_h264_maxframe=1 \
	    videoin_c0_s2_h264_maxvbrbitrate=10000000 \
	    videoin_c0_s2_mjpeg_ratecontrolmode=cbr \
	    videoin_c0_s2_mjpeg_bitrate=10000000 \
	    videoinpreview_maxexposure=5 \
	    videoinpreview_wdrc_mode=1 \
	    videoinpreview_maxgain=30 \
	    image_c0_saturation=100 \
	    image_c0_saturationpercent=59 \
	    image_c0_sharpness=100 \
	    image_c0_sharpnesspercent=0 \
	    image_c0_lowlightmode=0 \
	    imagepreview_c0_saturation=100 \
	    imagepreview_c0_saturationpercent=59 \
	    imagepreview_c0_sharpness=100 \
	    imagepreview_c0_sharpnesspercent=0 \
	    imagepreview_c0_lowlightmode=0 \
	    imagepreview_videoin_whitebalance=rbgain \
	    imagepreview_videoin_rgain=75 \
	    imagepreview_videoin_bgain=40 \
	    motion_c0_win_i0_enable=1 \
	    motion_c0_win_i0_name=ildkule \
	    motion_c0_win_i0_width=320 \
	    motion_c0_win_i0_height=171 \
	    motion_c0_win_i0_objsize=10 \
	    motion_c0_win_i0_sensitivity=90 \
	    audioin_c0_mute=1 \
	    server_i0_name=odroid \
	    server_i0_type=ns \
	    server_i0_ns_location=\\\\$ip\\meteor$n \
	    server_i0_ns_username=meteor \
	    server_i0_ns_passwd=XXPASSWORDXX \
	    recording_i0_name=1920x1080 \
	    recording_i0_enable=1 \
	    recording_i0_source=0 \
	    recording_i0_limitsize=0 \
	    recording_i0_cyclic=1 \
	    recording_i0_prefix=full_ \
	    recording_i0_reserveamount=16384 \
	    recording_i0_dest=0 \
	    recording_i0_adaptive_preevent=5 \
	    recording_i0_adaptive_postevent=5 \
	    recording_i1_name=800x600 \
	    recording_i1_enable=1 \
	    recording_i1_source=1 \
	    recording_i1_limitsize=0 \
	    recording_i1_cyclic=1 \
	    recording_i1_prefix=mini_ \
	    recording_i1_reserveamount=4194304 \
	    recording_i1_dest=0 \
	    recording_i1_adaptive_preevent=5 \
	    recording_i1_adaptive_postevent=5 \
	    ircutcontrol_mode=day \
	    ircutcontrol_bwmode=0 \
	    ircutcontrol_sensitivity=low \
	    layout_logo_default=0 \
	    layout_logo_link=http://norskmeteornettverk.no \
	    layout_logo_powerbyvvtk_hidden=1 \
	    disk_i0_cyclic_enabled=1 \
	    roi_c0_s0_home=0,0 \
	    roi_c0_s0_size=1920x1080 \
	    roi_c0_s1_home=0,0 \
	    roi_c0_s1_size=1920x1080 \
	    privacymask_c0_win_i0_enable=1 \
	    privacymask_c0_win_i0_name=clock \
	    privacymask_c0_win_i0_left=0 \
	    privacymask_c0_win_i0_top=237 \
	    privacymask_c0_win_i0_width=22 \
	    privacymask_c0_win_i0_height=3 \
            privacymask_c0_enable=1 \
            network_ftp_enable=1 \
	    ;
	do curl -u root:XXCAMPASSWORDXX "http://$i/cgi-bin/admin/setparam.cgi?$j"; sleep 0.3
	done
    fi

    if [[ "$cam" == "IP8172" ]]; then
	ncftpput -u root -p XXCAMPASSWORDXX $i /mnt/flash/www/pic /tmp/custom_logo.jpg 
	for j in \
	    system_hostname=XXSTATIONXX-$n \
 	    system_ntp=$ip \
	    system_timezoneindex=0 \
 	    system_daylight_auto_begintime=Disabled \
	    system_daylight_auto_endtime=Disabled \
	    system_updateinterval=3600 \
	    seamlessrecording_enable=1 \
	    network_rtsp_s0_audiotrack=-1 \
	    network_rtsp_s1_audiotrack=-1 \
	    network_rtsp_s2_audiotrack=-1 \
	    network_qos_dscp_video=63 \
	    videoin_cmosfreq=50 \
	    videoin_whitebalance=rbgain \
	    videoin_imprinttimestamp=1 \
	    videoin_maxexposure=10 \
	    videoin_options=crop \
	    videoin_c0_options=crop \
	    videoin_c0_preoptions=crop \
	    videoin_c0_crop_preview=0 \
	    videoin_c0_crop_position=-1,-1 \
	    videoin_c0_crop_size=2560x1920 \
	    videoin_c0_textonvideo_size=15 \
	    videoin_c0_textonvideo_position=bottom \
	    videoin_c0_cmosfreq=50 \
	    videoin_c0_whitebalance=rbgain \
	    videoin_c0_rgain=56 \
	    videoin_c0_bgain=20 \
	    videoin_c0_wdrc_mode=1 \
	    videoin_c0_maxgain=40 \
	    videoin_c0_imprinttimestamp=1 \
	    videoin_c0_maxexposure=10 \
	    videoin_c0_profile_i0_maxexposure=25 \
	    videoin_c0_s0_codectype=h264 \
	    videoin_c0_s0_resolution=2560x1920 \
	    videoin_c0_s0_mpeg4_intraperiod=250 \
	    videoin_c0_s0_h264_intraperiod=4000 \
	    videoin_c0_s0_h264_ratecontrolmode=vbr \
	    videoin_c0_s0_h264_quant=99 \
	    videoin_c0_s0_h264_qvalue=31 \
	    videoin_c0_s0_h264_qpercent=88 \
	    videoin_c0_s0_h264_bitrate=4000000 \
	    videoin_c0_s0_h264_maxframe=10 \
	    videoin_c0_s0_h264_profile=2 \
	    videoin_c0_s0_h264_maxvbrbitrate=6000000 \
	    videoin_c0_s0_h264_prioritypolicy=framerate \
	    videoin_c0_s1_codectype=h264 \
	    videoin_c0_s1_h264_maxframe=10 \
	    videoin_c0_s1_resolution=800x600 \
	    videoin_c0_s1_mpeg4_ratecontrolmode=vbr \
	    videoin_c0_s1_h264_intraperiod=4000 \
	    videoin_c0_s1_h264_quant=99 \
	    videoin_c0_s1_h264_qvalue=26 \
	    videoin_c0_s1_h264_bitrate=2000000 \
	    videoin_c0_s1_h264_profile=2 \
	    videoin_c0_s1_h264_maxvbrbitrate=20000000 \
	    videoin_c0_s1_h264_prioritypolicy=framerate \
	    videoin_c0_s2_h264_quant=99 \
	    videoin_c0_s2_h264_qvalue=24 \
	    videoin_c0_s2_h264_maxframe=1 \
	    videoin_c0_s2_h264_maxvbrbitrate=10000000 \
	    videoin_c0_s2_mjpeg_ratecontrolmode=cbr \
	    videoin_c0_s2_mjpeg_bitrate=10000000 \
	    videoinpreview_maxexposure=10 \
	    videoinpreview_wdrc_mode=1 \
	    videoinpreview_maxgain=40 \
	    image_c0_saturation=100 \
	    image_c0_saturationpercent=59 \
	    image_c0_sharpness=100 \
	    image_c0_sharpnesspercent=0 \
	    image_c0_lowlightmode=0 \
	    imagepreview_c0_saturation=100 \
	    imagepreview_c0_saturationpercent=59 \
	    imagepreview_c0_sharpness=100 \
	    imagepreview_c0_sharpnesspercent=0 \
	    imagepreview_c0_lowlightmode=0 \
	    imagepreview_videoin_whitebalance=rbgain \
	    imagepreview_videoin_rgain=64 \
	    imagepreview_videoin_bgain=16 \
	    motion_c0_win_i0_enable=1 \
	    motion_c0_win_i0_name=ildkule \
	    motion_c0_win_i0_width=320 \
	    motion_c0_win_i0_height=171 \
	    motion_c0_win_i0_objsize=10 \
	    motion_c0_win_i0_sensitivity=90 \
	    audioin_c0_mute=1 \
	    server_i0_name=odroid \
	    server_i0_type=ns \
	    server_i0_ns_location=\\\\$ip\\meteor$n \
	    server_i0_ns_username=meteor \
	    server_i0_ns_passwd=XXPASSWORDXX \
	    recording_i0_name=2560x1920 \
	    recording_i0_enable=1 \
	    recording_i0_source=0 \
	    recording_i0_limitsize=0 \
	    recording_i0_cyclic=1 \
	    recording_i0_prefix=full_ \
	    recording_i0_reserveamount=4194304 \
	    recording_i0_dest=0 \
	    recording_i0_adaptive_preevent=5 \
	    recording_i0_adaptive_postevent=5 \
	    recording_i1_name=800x600 \
	    recording_i1_enable=1 \
	    recording_i1_source=1 \
	    recording_i1_limitsize=0 \
	    recording_i1_cyclic=1 \
	    recording_i1_prefix=mini_ \
	    recording_i1_reserveamount=4194304 \
	    recording_i1_dest=0 \
	    recording_i1_adaptive_preevent=5 \
	    recording_i1_adaptive_postevent=5 \
	    ircutcontrol_mode=day \
	    ircutcontrol_bwmode=0 \
	    ircutcontrol_sensitivity=low \
	    layout_logo_default=0 \
	    layout_logo_link=http://norskmeteornettverk.no \
	    layout_logo_powerbyvvtk_hidden=1 \
	    disk_i0_cyclic_enabled=1 \
	    roi_c0_s0_home=0,0 \
	    roi_c0_s0_size=2560x1920 \
	    roi_c0_s1_home=0,0 \
	    roi_c0_s1_size=2560x1920 \
	    privacymask_c0_win_i0_enable=1 \
	    privacymask_c0_win_i0_name=clock \
	    privacymask_c0_win_i0_left=0 \
	    privacymask_c0_win_i0_top=237 \
	    privacymask_c0_win_i0_width=22 \
	    privacymask_c0_win_i0_height=3 \
            privacymask_c0_enable=1 \
            network_ftp_enable=1 \
	    ;
	do curl -u root:XXCAMPASSWORDXX "http://$i/cgi-bin/admin/setparam.cgi?$j"
	done
    fi

    if [[ "$cam" == "IP9171-HP" ]]; then
	ncftpput -u root -p XXCAMPASSWORDXX $i /mnt/flash/www/pic /tmp/custom_logo.jpg 
	for j in \
	    system_hostname=XXSTATIONXX-$n \
	    system_ntp=$ip \
	    system_timezoneindex=0 \
	    system_daylight_auto_begintime=Disabled \
	    system_daylight_auto_endtime=Disabled \
	    system_updateinterval=3600 \
	    seamlessrecording_enable=1 \
	    network_rtsp_s0_audiotrack=-1 \
	    network_rtsp_s1_audiotrack=-1 \
	    network_rtsp_s2_audiotrack=-1 \
	    network_qos_dscp_video=63 \
	    videoin_cmosfreq=60 \
	    videoin_whitebalance=rbgain \
	    videoin_imprinttimestamp=1 \
	    videoin_maxexposure=15\
	    videoin_options=crop \
	    videoin_c0_options=crop \
	    videoin_c0_preoptions=crop \
	    videoin_c0_crop_preview=0 \
	    videoin_c0_crop_position=-1,-1 \
	    videoin_c0_crop_size=2048x1536 \
	    videoin_c0_textonvideo_size=20 \
	    videoin_c0_textonvideo_position=bottom \
	    videoin_c0_cmosfreq=60 \
	    videoin_c0_whitebalance=rbgain \
	    videoin_c0_rgain=24 \
	    videoin_c0_bgain=24 \
	    videoin_c0_wdrc_mode=1 \
	    videoin_c0_wdrc_strength=3 \
	    videoin_c0_maxgain=85 \
	    videoin_c0_imprinttimestamp=1 \
	    videoin_c0_maxexposure=15 \
	    videoin_c0_profile_i0_maxexposure=15 \
	    videoin_c0_exposurelevel=6 \
	    videoin_c0_mingain=0 \
	    videoin_c0_color=1 \
	    videoin_c0_minexposure=16000 \
	    videoin_c0_maxexposure=15 \
	    videoin_c0_s0_codectype=h264 \
	    videoin_c0_s0_resolution=2048x1536 \
	    videoin_c0_s0_mpeg4_intraperiod=250 \
	    videoin_c0_s0_h264_intraperiod=4000 \
	    videoin_c0_s0_h264_ratecontrolmode=vbr \
	    videoin_c0_s0_h264_quant=99 \
	    videoin_c0_s0_h264_qvalue=30 \
	    videoin_c0_s0_h264_qpercent=88 \
	    videoin_c0_s0_h264_cbr_quant=5 \
	    videoin_c0_s0_h264_cbr_qpercent=70 \
	    videoin_c0_s0_h264_bitrate=8000000 \
	    videoin_c0_s0_h264_maxframe=30 \
	    videoin_c0_s0_h264_profile=2 \
	    videoin_c0_s0_h264_maxvbrbitrate=40000000 \
	    videoin_c0_s0_h264_prioritypolicy=framerate \
	    videoin_c0_s1_codectype=h264 \
	    videoin_c0_s1_h264_maxframe=30 \
	    videoin_c0_s1_resolution=800x600 \
	    videoin_c0_s1_mpeg4_ratecontrolmode=vbr \
	    videoin_c0_s1_h264_intraperiod=4000 \
	    videoin_c0_s1_h264_quant=99 \
	    videoin_c0_s1_h264_qvalue=26 \
	    videoin_c0_s1_h264_bitrate=2000000 \
	    videoin_c0_s1_h264_profile=2 \
	    videoin_c0_s1_h264_maxvbrbitrate=20000000 \
	    videoin_c0_s1_h264_prioritypolicy=framerate \
	    videoin_c0_s2_h264_quant=99 \
	    videoin_c0_s2_h264_qvalue=24 \
	    videoin_c0_s2_h264_maxframe=1 \
	    videoin_c0_s2_h264_maxvbrbitrate=10000000 \
	    videoin_c0_s2_mjpeg_ratecontrolmode=cbr \
	    videoin_c0_s2_mjpeg_bitrate=10000000 \
	    videoinpreview_maxexposure=15 \
	    videoinpreview_wdrc_mode=1 \
	    videoinpreview_maxgain=40 \
	    image_c0_saturation=100 \
	    image_c0_saturationpercent=67 \
	    image_c0_sharpness=100 \
	    image_c0_sharpnesspercent=0 \
	    image_c0_lowlightmode=0 \
	    imagepreview_c0_saturation=100 \
	    imagepreview_c0_saturationpercent=59 \
	    imagepreview_c0_sharpness=100 \
	    imagepreview_c0_sharpnesspercent=0 \
	    imagepreview_c0_lowlightmode=0 \
	    imagepreview_videoin_whitebalance=rbgain \
	    imagepreview_videoin_rgain=24 \
	    imagepreview_videoin_bgain=40 \
	    motion_c0_win_i0_enable=1 \
	    motion_c0_win_i0_name=ildkule \
	    motion_c0_win_i0_width=320 \
	    motion_c0_win_i0_height=171 \
	    motion_c0_win_i0_objsize=10 \
	    motion_c0_win_i0_sensitivity=90 \
	    audioin_c0_mute=1 \
	    server_i0_name=odroid \
	    server_i0_type=ns \
	    server_i0_ns_location=\\\\$ip\\meteor$n \
	    server_i0_ns_username=meteor \
	    server_i0_ns_passwd=XXPASSWORDXX \
	    recording_i0_name=2048x1536 \
	    recording_i0_enable=1 \
	    recording_i0_source=0 \
	    recording_i0_limitsize=0 \
	    recording_i0_cyclic=1 \
	    recording_i0_prefix=full_ \
	    recording_i0_reserveamount=4194304 \
	    recording_i0_begintime=00:00 \
	    recording_i0_endtime=24:00 \
	    recording_i0_dest=0 \
	    recording_i0_adaptive_preevent=5 \
	    recording_i0_adaptive_postevent=5 \
	    recording_i1_name=800x600 \
	    recording_i1_enable=1 \
	    recording_i1_source=1 \
	    recording_i1_limitsize=0 \
	    recording_i1_cyclic=1 \
	    recording_i1_prefix=mini_ \
	    recording_i1_reserveamount=4194304 \
	    recording_i1_dest=0 \
	    recording_i1_adaptive_preevent=5 \
	    recording_i1_adaptive_postevent=5 \
	    ircutcontrol_mode=day \
	    ircutcontrol_bwmode=0 \
	    ircutcontrol_sensitivity=low \
	    ircutcontrol_disableirled=1 \
	    ircutcontrol_enableextled=0 \
	    layout_logo_default=0 \
	    layout_logo_link=http://norskmeteornettverk.no \
	    layout_logo_powerbyvvtk_hidden=1 \
	    disk_i0_cyclic_enabled=1 \
	    roi_c0_s0_home=0,0 \
	    roi_c0_s0_size=2048x1536 \
	    roi_c0_s1_home=0,0 \
	    roi_c0_s1_size=2048x1536 \
	    privacymask_c0_win_i0_enable=1 \
	    privacymask_c0_win_i0_name=clock \
	    privacymask_c0_win_i0_polygon=0,233,45,233,45,237,0,237 \
            privacymask_c0_enable=1 \
	    videoin_c0_wdrc_mode=1 \
	    videoin_c0_wdrc_strength=1 \
	    videoin_c0_wdrpro_mode=0 \
	    image_c0_defog_mode=1 \
	    image_c0_defog_strength=100 \
	    videoin_c0_s0_h264_smartstream2_enable=0 \
	    videoin_c0_s0_h264_smartstream2_qualitypriority=3 \
	    videoin_c0_s0_h264_smartstream2_mode=autotracking \
	    videoin_irismode=fixed \
	    videoin_c0_irismode=fixed \
	    videoin_exposurelevel=6 \
	    videoin_maxexposure=15 \
	    videoin_minexposure=16000 \
	    videoin_c0_exposuremode=manual \
	    videoin_c0_piris_mode=manual \
	    videoin_c0_piris_position=1 \
	    image_c0_contrastpercent=51 \
	    image_c0_brightnesspercent=50 \
	    image_c0_dnr_strength=51 \
            network_ftp_enable=1 \
	    ;
	do curl -u root:XXCAMPASSWORDXX "http://$i/cgi-bin/admin/setparam.cgi?$j"
	done
	#curl -u root:XXCAMPASSWORDXX -s "http://$i:/cgi-bin/admin/remotefocus.cgi?function=focus&direction=direct&position=29"
    fi
done

rm -f /tmp/custom_logo.jpg
(echo -n $(date -u +%H%m)" "; /home/meteor/bin/sunriset $(grep latitude /etc/meteor.cfg | sed s/.*=//) $(grep longitude /etc/meteor.cfg | sed s/.*=//g) -6 | sed s/://g) | awk '{if ($1 > $2 || $1 < $3) { system("/root/bin/iris_night.sh") } else { system("/root/bin/iris_day.sh") } }'

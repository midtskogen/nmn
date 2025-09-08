#!/usr/bin/env python3

# Telnet root password: xmhdipc

from dvrip import DVRIPCam
from pprint import pprint

ips = [ "192.168.76.71", "192.168.76.72", "192.168.76.73", "192.168.76.74", "192.168.76.75", "192.168.76.76", "192.168.76.77" ]
#ips = [ "192.168.76.71" ]

for ip in ips:
    cam = DVRIPCam(ip, user='admin', password='')
    cam.login()
    cam.set_time()

    cam_settings = cam.get_info("Camera.Param")
    print(cam_settings)
#    cam_settings[0]['EsShutter'] = '0x00000002'
    cam_settings[0]['EsShutter'] = '0x00000000'
    cam_settings[0]['DayNightColor'] = '0x00000001'
    cam_settings[0]["GainParam"]["AutoGain"] = 1
    cam_settings[0]["GainParam"]["Gain"] = 50
    cam_settings[0]["BroadTrends"]["AutoGain"] = 0
    cam_settings[0]["BroadTrends"]["Gain"] = 50
    cam_settings[0]["ExposureParam"]["MostTime"] = '0x00010000'
    cam_settings[0]["ExposureTime"] = '0x00000100'
    cam_settings[0]["LowLuxMode"] = 0
    cam_settings[0]["LightRestrainLevel"] = 16
    cam_settings[0]['WhiteBalance'] = '0x00000000'
    cam_settings[0]['AutomaticAdjustment'] = 3
    cam_settings[0]['Day_nfLevel'] = 5
    cam_settings[0]['Night_nfLevel'] = 5
    cam_settings[0]['LowLuxMode'] = 0
    cam_settings[0]['Ldc'] = 1

    cam.set_info("Camera.Param", cam_settings)
    cam.set_info("Camera.ParamEx", cam_settings)
    
    enc_settings = cam.get_info("AVEnc.Encode")
    enc_settings[0]["MainFormat"][0]["Video"]["Resolution"] = '1080P'
    enc_settings[0]["MainFormat"][0]["Video"]["BitRate"] = 3072
    enc_settings[0]["MainFormat"][0]["Video"]["Quality"] = 6
    enc_settings[0]["MainFormat"][0]["Video"]["GOP"] = 6
    enc_settings[0]["ExtraFormat"][0]["Video"]["Resolution"] = 'HD1'
    enc_settings[0]["ExtraFormat"][1]["Video"]["Resolution"] = 'WSVGA'
    enc_settings[0]["ExtraFormat"][2]["Video"]["Resolution"] = 'WSVGA'
    cam.set_info("AVEnc.Encode", enc_settings)

    widget_settings = cam.get_info("AVEnc.VideoWidget")
    widget_settings[0]["ChannelTitle"]["Name"] = "NMN"
    widget_settings[0]["ChannelTitleAttribute"]["RelativePos"] = [8192, 8192, 0, 0]
    widget_settings[0]["ChannelTitleAttribute"]["EncodeBlend"] = False
    widget_settings[0]["Covers"][0]["EncodeBlend"] = True
    widget_settings[0]["Covers"][0]["RelativePos"] = [0, 7930, 1320, 8130]
#    widget_settings[0]["Covers"][0]["RelativePos"] = [0, 7930, 1540, 8140]
    widget_settings[0]["TimeTitleAttribute"]["BackColor"] = "0xFF808080"
    widget_settings[0]["TimeTitleAttribute"]["EncodeBlend"] = True
    widget_settings[0]["TimeTitleAttribute"]["FrontColor"] = "0xFFFFFFFF"
    widget_settings[0]["TimeTitleAttribute"]["PreviewBlend"] = True
    widget_settings[0]["TimeTitleAttribute"]["RelativePos"] = [0, 8192, 0, 0]
    widget_settings[0]["TimeTitleAttribute"]["RelativePos"] = [0, 8192, 0, 0]
    cam.set_info("AVEnc.VideoWidget", widget_settings)
    cam.reboot()

    
#[[{'Enable': True,
#   'TimeSection': '0 00:00:00-24:00:00',
#   'VideoColorParam': {'Acutance': 3848,
#                       'Brightness': 50,
#                       'Contrast': 50,
#                       'Gain': 0,
#                       'Hue': 50,
#                       'Saturation': 50,
#                       'Whitebalance': 128}},

exit()

settings = [ "fVideo.Tour", "fVideo.GUISet", "fVideo.WheelFunction", "fVideo.TVAdjust", "fVideo.AudioInFormat", "fVideo.Play", "fVideo.VideoOut", "fVideo.OSDWidget", "fVideo.Spot", "fVideo.Volume", "fVideo.LossShowStr", "fVideo.VideoOutPriority", "fVideo.VideoSeque", "fVideo.VoColorAdjust", "fVideo.OSDInfo", "fVideo.OsdLogo", "fVideo.VideoSignal", "fVideo.OEMChSeq", "fVideo.AudioSupportType", "Camera.Param", "Camera.ParamEx", "Camera.FishEye", "Camera.ClearFog", "Camera.MotorCtrl", "Camera.FishLensParam", "Camera.DistortionCorrect", "Camera.FishViCut", "Camera.WhiteLight", "AVEnc.CombineEncodeParam", "AVEnc.EncodeStaticParam", "AVEnc.Encode", "AVEnc.VideoWidget", "AVEnc.VideoColor", "AVEnc.CombineEncode", "AVEnc.CombineEncodeParam", "AVEnc.WaterMark", "AVEnc.EncodeStaticParamV2", "AVEnc.VideoColorCustom", "AVEnc.EncodeEx", "AVEnc.EncodeAddBeep" ]

for i in settings:
    print("\n\n" + i + ":")
    pprint(cam.get_info(i))
exit()
#pprint(cam.get_general_info())
#pprint(cam.get_system_info())
#pprint(cam.get_camera_info())
#pprint(cam.get_system_capabilities())
#pprint(cam.get_info("Camera"))
#pprint(cam.get_info("Simplify.Encode"))
#pprint(cam.get_info("NetWork.NetCommon"))
#exit()

# 'Param': [{'AeMeansure': 0,
#            'AeSensitivity': 5,
#            'ApertureMode': '0x00000000',
#            'AutomaticAdjustment': 3,
#            'BLCMode': '0x00000000',
#            'BroadTrends': {'AutoGain': 0, 'Gain': 50},
#            'CorridorMode': 0,
#            'DayNightColor': '0x00000000',
#            'Day_nfLevel': 3,
#            'Dis': 0,
#            'DncThr': 30,
#            'ElecLevel': 50,
#            'EsShutter': '0x00000000',
#            'ExposureParam': {'LeastTime': '0x00000100',
#                              'Level': 0,
#                              'MostTime': '0x00010000'},
#            'ExposureTime': '0x00000100',
#            'GainParam': {'AutoGain': 1, 'Gain': 50},
#            'IRCUTMode': 0,
#            'IrcutSwap': 0,
#            'Ldc': 0,
#            'LightRestrainLevel': 16,
#            'LowLuxMode': 0,
#            'Night_nfLevel': 3,
#            'PictureFlip': '0x00000000',
#            'PictureMirror': '0x00000000',
#            'PreventOverExpo': 0,
#            'RejectFlicker': '0x00000000',
#            'SoftPhotosensitivecontrol': 0,
#            'Style': 'type1',
#            'WhiteBalance': '0x00000000'}],
# 'ParamEx': [{'AeMeansure': 0,
#              'AutomaticAdjustment': 3,
#              'BroadTrends': {'AutoGain': 0, 'Gain': 50},
#              'CorridorMode': 0,
#              'Dis': 0,
#              'EsShutter': '0x00000000',
#              'ExposureTime': '0x00000100',
#              'Ldc': 0,
#              'LightRestrainLevel': 16,
#              'LowLuxMode': 0,
#              'PreventOverExpo': 0,
#              'SoftPhotosensitivecontrol': 0,
#              'Style': 'type1'}],

#pprint(cam.get_info("Camera.Param"))

#cam_settings = cam.get_info("Camera.ParamEx")
#cam_settings[0]['BroadTrends']['AutoGain'] = 1
#cam.set_info("Camera.ParamEx", cam_settings)

#pprint(cam_settings)
#sleep(5)
#cam_settings[0]['WhiteBalance'] = '0x00000000'
#cam.set_info("Camera.Param", cam_settings)

cam_text = cam.get_info("fVideo.OSDInfo")
#pprint(cam_text)
cam_text["OSDInfo"][0]["Info"] = ["", None]
cam_text["OSDInfo"][0]["OSDInfoWidget"]["EncodeBlend"] = True
cam_text["OSDInfo"][0]["OSDInfoWidget"]["BackColor"]= '0x80000000'
cam_text["OSDInfo"][0]["OSDInfoWidget"]["FrontColor"]= '0xF0FFFF00'
cam_text["OSDInfo"][0]["OSDInfoWidget"]["RelativePos"] = [0,0,0,0]
cam.set_info("fVideo.OSDInfo", cam_text)

cam_gui = cam.get_info("fVideo.GUISet")
cam_gui["TimeTitleEnable"] = True
cam.set_info("fVideo.GUISet", cam_gui)

#cam_widget = cam.get_info("fVideo.OSDWidget")
#pprint(cam_widget)
#cam_widget[0]["AlarmInfo"]["EncodeBlend"] = True
#cam.set_info("fVideo.OSDWidget", cam_widget)

# 'OSDWidget': [{'AlarmInfo': {'BackColor': '0x60000000',
#                              'EncodeBlend': True,
#                              'FrontColor': '0xF0FFFFFF',
#                              'PreviewBlend': True,
#                              'RelativePos': [570, 2048, 255, 24]}}],

# 'OsdLogo': {'BgTrans': 16, 'Enable': 1, 'FgTrans': 96, 'Left': 80, 'Top': 80},


#cam.channel_title([""])

pprint(cam.get_info("PreviewFunction"))

enc_settings = cam.get_info("Simplify.Encode")
#enc_settings[0]["MainFormat"]["Video"]["BitRate"] = 6144 # 6144
#enc_settings[0]["MainFormat"]["Video"]["FPS"] = 25
enc_settings[0]["MainFormat"]["Video"]["GOP"] = 4 # 6144
#del enc_settings[0]["MainFormat"]["Video"]["Bitrate"]
cam.set_info("Simplify.Encode", enc_settings)
#pprint(enc_settings)

# Disconnect
cam.close()

import numpy as np
from turbine_class_V1 import *

dt= 0.1
time = 600
hs = 5
tp = 11
sea_floor=300
floater_bottom=30


V_10 = 9
I = 0.15
l = 340.2
z_hub = 119

dict_testing = {"time_span":time,
                   "time_change":dt,
                   "wave_height":hs,
                   "wave_period":tp,
                   "sea_floor":sea_floor,
                   "floater_bottom":floater_bottom,
                   "wind_mean":V_10,
                   "turb_intens":I,
                   "hub_height":z_hub}

windy_wave_class = Windy_wavy(dict_testing)




windy_wave_class.run_windy_wavy()

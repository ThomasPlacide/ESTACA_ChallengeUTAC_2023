# ---------------- SAMPLE -------------------
# This sample shows how to create an ioelt of type RealObjects from scratch.
# Then we fill its fields.

import rtmaps.types
import numpy as np
from rtmaps.base_component import BaseComponent # base class
import rtmaps.core as rt
import rtmaps.reading_policy


# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
        self.force_reading_policy(rtmaps.reading_policy.SAMPLING)
    
    # All inputs, outputs and properties MUST be created in the
    # Dynamic() function.
    def Dynamic(self):
        self.add_output("out", rtmaps.types.REAL_OBJECT)  # define output

# Birth() will be called once at diagram execution startup
    def Birth(self):
        self.i = 0

# Core() is called every time you have a new input, depending
# on the reading policy you have chosen
    def Core(self):
        out = rtmaps.types.Ioelt()
        out.data = []
        out.data.append(rtmaps.types.RealObject())
        out.data[0].kind = 0   # 0 = Vehicle, 1 = Sign, 2 = Tree, 3 = Custom
        out.data[0].id = 0
        out.data[0].x = 1.0
        out.data[0].y = 1.0
        out.data[0].z = 0.0
        out.data[0].color = (0 << 16) + (0 << 8) + 255 # BGR : so Red here
        out.data[0].misc1 = 0
        out.data[0].misc2 = 0
        out.data[0].misc3 = 0

        out.data[0].data = rtmaps.types.Vehicle()  # RealObject of type Vehicle
        out.data[0].data.kind = 3  # 0 = Car, 1 = Bus, 2 = Truck, 3 = Bike, 4 = Motorcycle
        out.data[0].data.theta = 0.0
        out.data[0].data.speed = 0.0
        out.data[0].data.width = 10.0
        out.data[0].data.height = 6.0
        out.data[0].data.length = 6.0
        out.data[0].data.model = 0
        out.data[0].data.braking = False
        out.data[0].data.confidence = 0.0
        out.data[0].data.dx = 0.0
        out.data[0].data.dy = 0.0
        out.data[0].data.dz = 0.0

        self.outputs["out"].write(out)  # and write it to the output

# Death() will be called once at diagram execution shutdown
    def Death(self):
        rt.report_info("Death")

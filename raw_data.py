# ===========================================================================
# Copyright (C) 2024 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import pprint
import numpy as np
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp


config = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=307.325e-3,  # Frame repetition time
    chirp_repetition_time_s=500e-6,  # Chirp repetition time
    num_chirps=16,  # chirps per frame
    tdm_mimo=False,  # set True to enable MIMO mode, which is only valid for sensors with 2 Tx antennas
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=59e9,  # start RF frequency, where Tx is ON
        end_frequency_Hz=61e9,  # stop RF frequency, where Tx is OFF
        sample_rate_Hz=2e6,  # ADC sample rate
        num_samples=128,  # samples per chirp
        rx_mask=1,  # RX mask is a 4-bit, each bit set enables that RX e.g. [1,3,7,15]
        tx_mask=1,  # TX antenna mask is a 2-bit (use value 3 for MIMO)
        tx_power_level=31,  # TX power level of 31
        lp_cutoff_Hz=500000,  # Anti-aliasing filter cutoff frequency, select value from data-sheet
        hp_cutoff_Hz=80000,  # High-pass filter cutoff frequency, select value from data-sheet
        if_gain_dB=30,  # IF-gain
    ),
)

# open device: The device will be closed at the end of the block. Instead of
# the with-block you can also use:
#   device = DeviceFmcw()
# However, the with block gives you better control when the device is closed.
with DeviceFmcw() as device:
    print("Radar SDK Version: " + get_version())
    print("UUID of board: " + device.get_board_uuid())
    print("Sensor: " + str(device.get_sensor_type()))
    pp = pprint.PrettyPrinter()

    # configure device
    sequence = device.create_simple_sequence(config)
    device.set_acquisition_sequence(sequence)

    # get metrics from a chirp config and print them
    chirp_loop = sequence.loop.sub_sequence.contents
    metrics = device.metrics_from_sequence(chirp_loop)
    pp.pprint(metrics)

    # Print the complete acquisition sequence
    pp.pprint(create_dict_from_sequence(sequence))

    # save device configuration to a register file
    device.save_register_file("exported_registers.txt")

    # Fetch a number of frames
    for frame_number in range(10):
        frame_contents = device.get_next_frame()

        for frame in frame_contents:
            num_rx = np.shape(frame)[0]
            print(np.shape(frame))
            # Do some processing with the obtained frame.
            # In this example we just dump it into the console
            print("Frame " + format(frame_number) + ", num_antennas={}".format(num_rx))

            for iAnt in range(num_rx):
                mat = frame[iAnt, :, :]
                print("Antenna", iAnt, "\n", mat)

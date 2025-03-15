import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp
from helpers.DigitalBeamForming import DigitalBeamForming
from helpers.DopplerAlgo import DopplerAlgo

def num_rx_antennas_from_rx_mask(rx_mask):
    c = 0
    for i in range(32):
        if rx_mask & (1 << i):
            c += 1
    return c

class AngleLivePlot:
    def __init__(self, max_angle_degrees):
        self._fig, self._ax = plt.subplots()
        self._fig.canvas.manager.set_window_title("Dominant Angle vs. Time")
        self._fig.canvas.mpl_connect('close_event', self.close)

        self._is_window_open = True
        self.max_angle_degrees = max_angle_degrees

        self.angles = []
        self.times = []
        self.line, = self._ax.plot([], [], 'b-o')
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Dominant Angle (degrees)")
        self._ax.set_ylim(-self.max_angle_degrees, self.max_angle_degrees)
        self._ax.grid(True)

        plt.ion()
        plt.show()

    def update(self, timestamp: float, angle_degrees: float):
        if not self._is_window_open:
            return

        self.times.append(timestamp)
        self.angles.append(angle_degrees)

        self.line.set_xdata(self.times)
        self.line.set_ydata(self.angles)

        self._ax.set_xlim(self.times[0], max(self.times[0] + 10, timestamp))

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self, event=None):
        if not self.is_closed():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_closed(self):
        return not self._is_window_open

if __name__ == '__main__':
    num_beams = 27
    max_angle_degrees = 40

    output_path = r"C:\Users\Raj Patel\Desktop\radar_sdk\Data_Store\angle_data.xlsx"
    
    config = FmcwSimpleSequenceConfig(
        frame_repetition_time_s=0.15,
        chirp_repetition_time_s=0.0005,
        num_chirps=128,
        tdm_mimo=False,
        chirp=FmcwSequenceChirp(
            start_frequency_Hz=60e9,
            end_frequency_Hz=61.5e9,
            sample_rate_Hz=1e6,
            num_samples=64,
            rx_mask=5,
            tx_mask=1,
            tx_power_level=31,
            lp_cutoff_Hz=500000,
            hp_cutoff_Hz=80000,
            if_gain_dB=33,
        )
    )

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        sequence = device.create_simple_sequence(config)
        device.set_acquisition_sequence(sequence)

        chirp_loop = sequence.loop.sub_sequence.contents
        metrics = device.metrics_from_sequence(chirp_loop)
        pprint.pprint(metrics)

        max_range_m = metrics.max_range_m

        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        num_rx_antennas = num_rx_antennas_from_rx_mask(chirp.rx_mask)

        doppler = DopplerAlgo(config.chirp.num_samples, config.num_chirps, num_rx_antennas)
        dbf = DigitalBeamForming(num_rx_antennas, num_beams=num_beams, max_angle_degrees=max_angle_degrees)

        plot = AngleLivePlot(max_angle_degrees)

        timestamps = []
        angles = []
        start_time = time.time()

        while not plot.is_closed():
            frame_contents = device.get_next_frame()
            frame = frame_contents[0]

            rd_spectrum = np.zeros((config.chirp.num_samples, 2 * config.num_chirps, num_rx_antennas), dtype=complex)
            beam_range_energy = np.zeros((config.chirp.num_samples, num_beams))

            for i_ant in range(num_rx_antennas):
                mat = frame[i_ant, :, :]
                dfft_dbfs = doppler.compute_doppler_map(mat, i_ant)
                rd_spectrum[:, :, i_ant] = dfft_dbfs

            rd_beam_formed = dbf.run(rd_spectrum)

            for i_beam in range(num_beams):
                doppler_i = rd_beam_formed[:, :, i_beam]
                beam_range_energy[:, i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(num_beams)

            max_energy = np.max(beam_range_energy)
            scale = 150
            beam_range_energy = scale * (beam_range_energy / max_energy - 1)

            _, idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
            angle_degrees = np.linspace(-max_angle_degrees, max_angle_degrees, num_beams)[idx]

            timestamp = time.time() - start_time
            plot.update(timestamp, angle_degrees)
            timestamps.append(timestamp)
            angles.append(angle_degrees)

        plot.close()

        df = pd.DataFrame({"Time (s)": timestamps, "Angle (degrees)": angles})
        df.to_excel(output_path, index=False)
        print(f"Data saved to {output_path}")

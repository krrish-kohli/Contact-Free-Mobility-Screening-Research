import os
import argparse
import sys
import time
import pandas as pd
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DistanceAlgo import *

# Define the save path for the Excel file
SAVE_PATH = r"C:\Users\Raj Patel\Desktop\radar_sdk\Data_Store\distance_data.xlsx"


class DistanceWindow(QtWidgets.QMainWindow):
    def __init__(self, frate=10, nframes=0, parent=None):
        super(DistanceWindow, self).__init__(parent)
        self._distance = []
        self._times = []
        self._mainbox = QtWidgets.QWidget()
        self.setCentralWidget(self._mainbox)
        self._mainbox.setLayout(QtWidgets.QVBoxLayout())

        self._detection = QtWidgets.QLabel()
        self._mainbox.layout().addWidget(self._detection)

        self._canvas = pg.GraphicsLayoutWidget()
        self._mainbox.layout().addWidget(self._canvas)

        self._label = QtWidgets.QLabel()
        self._mainbox.layout().addWidget(self._label)

        self._button = QtWidgets.QPushButton("Save & Clear")
        self._button.clicked.connect(self._save_and_clear)
        self._mainbox.layout().addWidget(self._button)

        self._plot = self._canvas.addPlot()
        self._plot.addLegend()
        self._plot.setLabels(bottom="time [s]", left="distance [m]")
        self._line_distance = self._plot.plot(pen='g', name="distance")

        self._counter = 0
        self._start_time = time.time()

    def update_distance(self, distance):
        self._counter += 1
        self._distance.append(distance)
        self._times.append(time.time() - self._start_time)
        self._line_distance.setData(self._times, self._distance)

    def _save_and_clear(self):
        self.save_to_excel()
        self._clear()

    def _clear(self):
        self._distance.clear()
        self._times.clear()
        self._counter = 0
        self._start_time = time.time()

    def save_to_excel(self):
        if not self._distance:
            print("No data to save.")
            return
        
        df = pd.DataFrame({'Time (s)': self._times, 'Distance (m)': self._distance})
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

        if os.path.exists(SAVE_PATH):
            existing_df = pd.read_excel(SAVE_PATH)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_excel(SAVE_PATH, index=False)
        print(f"Data saved to {SAVE_PATH}")


class DistanceUpdaterTimer(QtCore.QTimer):
    def __init__(self, device, window, dist_algo):
        super(QtCore.QTimer, self).__init__()
        self._device = device
        self._window = window
        self._dist_algo = dist_algo
        self.timeout.connect(self._update_distance)

    def _update_distance(self):
        frame_contents = self._device.get_next_frame()
        frame = frame_contents[0]
        antenna_samples = frame[i_ant, :, :]
        distance_peak_m, distance_data = self._dist_algo.compute_distance(antenna_samples)
        self._window.update_distance(distance_peak_m)

    def start(self):
        super().start(100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Displays distance plot from Radar Data''')
    parser.add_argument('-f', '--frate', type=int, default=5, help="frame rate in Hz, default 5")
    args = parser.parse_args()

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        i_ant = 0  
        num_rx_antennas = 1

        metrics = FmcwMetrics(
            range_resolution_m=0.05,
            max_range_m=1.6,
            max_speed_m_s=3,
            speed_resolution_m_s=0.2,
            center_frequency_Hz=60_750_000_000,
        )

        sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
        sequence.loop.repetition_time_s = 1 / args.frate  
        chirp_loop = sequence.loop.sub_sequence.contents
        device.sequence_from_metrics(metrics, chirp_loop)

        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        chirp.sample_rate_Hz = 1_000_000
        chirp.rx_mask = (1 << num_rx_antennas) - 1
        chirp.tx_mask = 1
        chirp.tx_power_level = 31
        chirp.if_gain_dB = 33
        chirp.lp_cutoff_Hz = 500000
        chirp.hp_cutoff_Hz = 80000

        device.set_acquisition_sequence(sequence)

        algo = DistanceAlgo(chirp, chirp_loop.loop.num_repetitions)

        app = QtWidgets.QApplication(sys.argv)
        distance_window = DistanceWindow(args.frate)
        distance_window.show()

        timer = DistanceUpdaterTimer(device, distance_window, algo)
        timer.start()

        app.exec_()

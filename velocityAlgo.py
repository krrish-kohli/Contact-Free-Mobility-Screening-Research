# velocityAlgo.py

# ===========================================================================
# [License Header]
# ===========================================================================

import argparse
import sys
import time
import ctypes  # Required for c_bool

from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from helpers.DopplerAlgo import DopplerAlgo


# -------------------------------------------------
# Presentation 
# -------------------------------------------------
class VelocityWindow(QtWidgets.QMainWindow):
    """GUI window showing velocities"""

    def __init__(self, frate=10, nframes=0, parent=None):
        super(VelocityWindow, self).__init__(parent)
        self._velocity = []
        self._times = []

        # Create GUI Elements
        self._mainbox = QtWidgets.QWidget()
        self.setCentralWidget(self._mainbox)
        self._mainbox.setLayout(QtWidgets.QVBoxLayout())

        self._detection = QtWidgets.QLabel()
        self._mainbox.layout().addWidget(self._detection)

        self._canvas = pg.GraphicsLayoutWidget()
        self._mainbox.layout().addWidget(self._canvas)

        self._label = QtWidgets.QLabel()
        self._mainbox.layout().addWidget(self._label)

        self._button = QtWidgets.QPushButton("Clear")
        self._button.clicked.connect(self._clear)
        self._mainbox.layout().addWidget(self._button)

        self._plot = self._canvas.addPlot()
        self._plot.addLegend()
        self._plot.setLabels(bottom="Time [s]", left="Velocity [m/s]")
        self._line_velocity = self._plot.plot(pen='r', name="Velocity")

        self._counter = 0
        self._start_time = time.time()

    def update_velocity(self, velocity):
        """Update the velocity plot with a new velocity value"""
        self._counter += 1
        self._velocity.append(velocity)
        self._times.append(time.time() - self._start_time)
        self._line_velocity.setData(self._times, self._velocity)

    def _clear(self):
        """Clear the velocity plot"""
        self._velocity.clear()
        self._times.clear()
        self._counter = 0
        self._start_time = time.time()
        self._line_velocity.clear()


# Acquisition in QT timer + Presentation
# -------------------------------------------------
class VelocityUpdaterTimer(QtCore.QTimer):
    """Timer to periodically update velocity measurements"""

    def __init__(self, device, window, velocity_algo, chirp_duration, wavelength, i_ant=0):
        super(VelocityUpdaterTimer, self).__init__()
        self._device = device
        self._window = window
        self._velocity_algo = velocity_algo
        self._i_ant = i_ant

        # Set Doppler resolution based on radar parameters
        self._velocity_algo.set_doppler_resolution(chirp_duration, wavelength)

        # Connect the timeout signal to the update method
        self.timeout.connect(self._update_velocity)

    def _update_velocity(self):
        """Fetch the next frame, compute velocity, and update the GUI"""
        try:
            frame_contents = self._device.get_next_frame()
            frame = frame_contents[0]  # Assuming single frame
            antenna_samples = frame[self._i_ant, :, :]  # Shape: num_chirps_per_frame x num_samples

            velocity, doppler_map = self._velocity_algo.compute_velocity(antenna_samples, self._i_ant)
            self._window.update_velocity(velocity)
        except Exception as e:
            print(f"Error updating velocity: {e}")

    def start_timer(self, interval_ms=100):
        """Start the timer with the specified interval"""
        self.start(interval_ms)


# -------------------------------------------------
# Main logic
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='''Displays velocity plot from Radar Data''')
    parser.add_argument('-f', '--frate', type=int, default=5, help="Frame rate in Hz, default 5")
    args = parser.parse_args()

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print("Sensor: " + str(device.get_sensor_type()))

        i_ant = 0  # Use only 1st RX antenna
        num_rx_antennas = 1

        metrics = FmcwMetrics(
            range_resolution_m=0.05,
            max_range_m=1.6,
            max_speed_m=3,
            speed_resolution_m=0.2,
            center_frequency_Hz=60_750_000_000,
        )

        # Create acquisition sequence based on metrics parameters
        sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
        sequence.loop.repetition_time_s = 1 / args.frate  # Set frame repetition time

        # Convert metrics into chirp loop parameters
        chirp_loop = sequence.loop.sub_sequence.contents

        # **CORRECTION STARTS HERE**
        # Set the 'round_to_power_of_2' parameter as required.
        # Set to False unless you specifically need to round parameters to the nearest power of 2.
        round_to_power_of_2 = True  # Change to True if required

        # **Pass the 'round_to_power_of_2' parameter correctly**
        try:
            # Convert the Python bool to ctypes.c_bool
            round_to_power_of_2_ctypes = ctypes.c_bool(round_to_power_of_2)
            device.sequence_from_metrics(metrics, chirp_loop, round_to_power_of_2_ctypes)
        except Exception as e:
            print(f"Error in sequence_from_metrics: {e}")
            sys.exit(1)
        # **CORRECTION ENDS HERE**

        # Set remaining chirp parameters which are not derived from metrics
        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        chirp.sample_rate_Hz = 1_000_000
        chirp.rx_mask = (1 << num_rx_antennas) - 1
        chirp.tx_mask = 1
        chirp.tx_power_level = 31
        chirp.if_gain_dB = 33
        chirp.lp_cutoff_Hz = 500_000
        chirp.hp_cutoff_Hz = 80_000

        # **Important**: After setting chirp parameters, update the chirp loop
        # to ensure that the changes are applied.
        try:
            device.upload_sequence(sequence)
        except Exception as e:
            print(f"Error uploading sequence: {e}")
            sys.exit(1)

        # Calculate radar parameters for Doppler resolution
        wavelength = 3e8 / metrics.center_frequency_Hz  # Speed of light divided by center frequency
        chirp_duration = chirp.num_samples / chirp.sample_rate_Hz  # Total duration of chirp in seconds

        # Initialize Doppler algorithm
        velocity_algo = DopplerAlgo(
            num_samples=chirp.num_samples,
            num_chirps_per_frame=chirp_loop.loop.num_repetitions,
            num_ant=num_rx_antennas
        )

        # Initialize and start the Qt application
        app = QtWidgets.QApplication(sys.argv)
        velocity_window = VelocityWindow(args.frate)
        velocity_window.show()

        # Initialize and start the velocity updater timer
        timer = VelocityUpdaterTimer(
            device=device,
            window=velocity_window,
            velocity_algo=velocity_algo,
            chirp_duration=chirp_duration,
            wavelength=wavelength,
            i_ant=i_ant
        )
        timer.start_timer(interval_ms=100)  # Update every 100 ms

        # Execute the Qt application
        sys.exit(app.exec_())


if __name__ == '__main__':
    main()

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from scipy import constants as C

from ifxradarsdk import get_version_full
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
from ifxradarsdk.common.exceptions import ErrorFrameAcquisitionFailed

# Helpers from this repo
from helpers.DistanceAlgo import DistanceAlgo
from helpers.DopplerAlgo import DopplerAlgo
from helpers.DigitalBeamForming import DigitalBeamForming


class RadarProcessor:
    """Owns the device, algorithms, and all data buffers."""
    def __init__(self, config=None, data_dir=None):
        self.config = config or {
            'frame_rate': 5,                 # Hz
            'num_chirps': 128,
            'num_samples': 256,
            'range_resolution_m': 0.05,      # ~5 cm -> SDK picks B
            'max_range_m': 6.0,
            'max_speed_m_s': 3.0,
            'speed_resolution_m_s': 0.1,
            'max_angle_degrees': 60,
            'num_beams': 31,
        }
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), "radar_data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Connect device
        self.device = DeviceFmcw()
        info = self.device.get_sensor_information()
        self.num_rx_antennas = int(info.get("num_rx_antennas", 1))

        print(f"Radar SDK: {get_version_full()}")
        print(f"Sensor: {self.device.get_sensor_type()}  | RX antennas: {self.num_rx_antennas}")

        # Build sequence from metrics (let SDK choose Fs, N, B, T, etc.)
        self._configure_device()
        self._initialize_algorithms()

        # Time-series (combined for plotting)
        self.frame_index = 0
        self.start_time = time.time()
        self.timestamps = []
        self.distances = []
        self.velocities = []
        self.angles = []

        # Per-sensor logs (first three sensors as requested)
        self.s1_distances, self.s1_velocities, self.s1_powers = [], [], []
        self.s2_distances, self.s2_velocities, self.s2_powers = [], [], []
        self.s3_distances, self.s3_velocities, self.s3_powers = [], [], []

    # ---- Device setup -----------------------------------------------------

    def reset(self):
        """Clear time-series buffers for a new trial."""
        self.frame_index = 0
        self.start_time = time.time()
        self.timestamps = []
        self.distances = []
        self.velocities = []
        self.angles = []

        self.s1_distances, self.s1_velocities, self.s1_powers = [], [], []
        self.s2_distances, self.s2_velocities, self.s2_powers = [], [], []
        self.s3_distances, self.s3_velocities, self.s3_powers = [], [], []

    def _configure_device(self):
        m = FmcwMetrics(
            range_resolution_m=self.config['range_resolution_m'],
            max_range_m=self.config['max_range_m'],
            max_speed_m_s=self.config['max_speed_m_s'],
            speed_resolution_m_s=self.config['speed_resolution_m_s'],
            center_frequency_Hz=60_750_000_000,  # 60.75 GHz
        )

        seq = self.device.create_simple_sequence(FmcwSimpleSequenceConfig())
        # Frame repetition time controls frame_rate
        seq.loop.repetition_time_s = 1.0 / float(self.config['frame_rate'])

        chirp_loop = seq.loop.sub_sequence.contents
        self.device.sequence_from_metrics(m, chirp_loop)

        # Low-level chirp tweaks (do NOT override sample_rate here)
        chirp = chirp_loop.loop.sub_sequence.contents.chirp
        # Ensure ADC sample rate is set (Hz)
        if not getattr(chirp, "sample_rate_Hz", 0):
            chirp.sample_rate_Hz = 1_000_000  # 1 MHz

        self.config['sample_rate_Hz'] = chirp.sample_rate_Hz

        chirp.rx_mask = (1 << self.num_rx_antennas) - 1
        chirp.tx_mask = 1
        chirp.tx_power_level = 31
        chirp.if_gain_dB = 33
        chirp.lp_cutoff_Hz = 500_000
        chirp.hp_cutoff_Hz = 1_000  # lower HPF so near targets are visible

        # Back-fill config with actual values that SDK decided
        self.config['num_chirps'] = chirp_loop.loop.num_repetitions
        self.config['num_samples'] = chirp.num_samples

        self.chirp = chirp
        self.metrics = m
        self.device.set_acquisition_sequence(seq)

        # Precompute Doppler scaling terms (we'll compute velocity ourselves)
        self._lambda = C.c / float(self.metrics.center_frequency_Hz)
        self._PRF = float(self.config['frame_rate'] * self.config['num_chirps'])

    def _initialize_algorithms(self):
        self.range_algo = DistanceAlgo(self.chirp, self.config['num_chirps'])
        self.doppler_algo = DopplerAlgo(self.config['num_samples'], self.config['num_chirps'], self.num_rx_antennas)
        self.dbf = DigitalBeamForming(self.num_rx_antennas, self.config['num_beams'], self.config['max_angle_degrees'])

        # Range–Doppler cube for beamforming [ranges x doppler_bins x rx]
        Ndop = 2 * self.config['num_chirps']
        self.rd_spectrum = np.zeros((self.config['num_samples'], Ndop, self.num_rx_antennas), dtype=np.complex128)

    # ---- Processing -------------------------------------------------------

    def _doppler_velocity_from_map(self, rd_map, range_bin, gate_half_width=2):
        """Given an RD map (N_ranges x Ndop), return velocity (m/s) at the gated range."""
        Ndop = rd_map.shape[1]
        zero = Ndop // 2
        # Range gate
        lo = max(0, int(range_bin) - gate_half_width)
        hi = min(rd_map.shape[0], int(range_bin) + gate_half_width + 1)
        spectrum = np.abs(rd_map[lo:hi, :]).mean(axis=0)
        peak_idx = int(np.argmax(spectrum))
        doppler_bin = peak_idx - zero
        v_per_bin = (self._PRF / Ndop) * (self._lambda / 2.0)
        return float(doppler_bin * v_per_bin)

    def process_frame(self):
        """Grab one frame, compute per-sensor distance/velocity (+power) and a single angle."""
        # Acquire
        frame_contents = self.device.get_next_frame()
        frame = frame_contents[0]  # shape: [rx, chirps, samples]

        # Time
        t = time.time() - self.start_time
        self.timestamps.append(t)

        per_ranges, per_vels, per_powers = [], [], []

        # Per-antenna processing
        for i_ant in range(self.num_rx_antennas):
            ant = frame[i_ant, :, :]  # [num_chirps x num_samples]

            # Distance (using helper)
            dist_m, dist_spec = self.range_algo.compute_distance(ant)
            rb = int(round(dist_m / self.range_algo.range_bin_length))
            rb = max(0, min(rb, self.config['num_samples'] - 1))

            # Doppler map for this antenna; compute velocity ourselves (correct zero & scaling)
            rd_map = self.doppler_algo.compute_doppler_map(ant, i_ant)
            vel_m_s = self._doppler_velocity_from_map(rd_map, rb)

            # Save into RD cube for DBF (beamforming)
            self.rd_spectrum[:, :, i_ant] = rd_map

            # Simple power metric at detected range bin
            power_db = 20.0 * np.log10(dist_spec[rb] + 1e-12)

            per_ranges.append(float(dist_m))
            per_vels.append(float(vel_m_s))
            per_powers.append(float(power_db))

        # Angle via DBF (one angle per frame across sensors)
        rd_bf = self.dbf.run(self.rd_spectrum)  # [ranges x Ndop x beams]
        # Energy per beam (sum over doppler & ranges)
        beam_energy = np.linalg.norm(rd_bf, axis=1)  # [ranges x beams] -> sum over doppler already by norm
        beam_energy = beam_energy.sum(axis=0)        # [beams]
        angle_axis = np.linspace(-self.config['max_angle_degrees'], self.config['max_angle_degrees'],
                                 self.config['num_beams'])
        angle_deg = float(angle_axis[int(np.argmax(beam_energy))])

        # Combined values for on-screen plots (simple mean of sensors present)
        self.distances.append(float(np.nanmean(per_ranges)) if per_ranges else np.nan)
        self.velocities.append(float(np.nanmean(per_vels)) if per_vels else np.nan)
        self.angles.append(angle_deg)

        # Keep first three sensors in per-sensor logs (pad with NaN if fewer)
        def pick(a, i): return a[i] if i < len(a) else float('nan')
        self.s1_distances.append(pick(per_ranges, 0)); self.s1_velocities.append(pick(per_vels, 0)); self.s1_powers.append(pick(per_powers, 0))
        self.s2_distances.append(pick(per_ranges, 1)); self.s2_velocities.append(pick(per_vels, 1)); self.s2_powers.append(pick(per_powers, 1))
        self.s3_distances.append(pick(per_ranges, 2)); self.s3_velocities.append(pick(per_vels, 2)); self.s3_powers.append(pick(per_powers, 2))

        self.frame_index += 1

        return {
            'time': t,
            'distance': self.distances[-1],
            'velocity': self.velocities[-1],
            'angle': angle_deg,
        }

    # ---- Persistence ------------------------------------------------------

    def save_data(self, filename=None):
        """Save to Excel with columns:
        time | s1_distance | s1_velocity | s1_power | s2_distance | s2_velocity | s2_power |
        s3_distance | s3_velocity | s3_power | angle_deg
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"radar_data_{timestamp}.xlsx"
        file_path = os.path.join(self.data_dir, filename)

        df = pd.DataFrame({
            'time': self.timestamps,
            's1_distance': self.s1_distances, 's1_velocity': self.s1_velocities, 's1_power': self.s1_powers,
            's2_distance': self.s2_distances, 's2_velocity': self.s2_velocities, 's2_power': self.s2_powers,
            's3_distance': self.s3_distances, 's3_velocity': self.s3_velocities, 's3_power': self.s3_powers,
            'angle_deg': self.angles,
        })
        try:
            df.to_excel(file_path, index=False)
            print(f"Data saved to {file_path}")
        except Exception as e:
            # Fallback to CSV if Excel writer is missing
            csv_path = file_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"Excel save failed ({e}); saved CSV to {csv_path}")
        return file_path

    def close(self):
        # try:
        #     del self.device
        # except Exception:
        pass


class RadarGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FMCW Radar GUI (per-sensor logging)")
        self.processing_active = False
        self.end_time = None

        # Instantiate processor
        self.processor = RadarProcessor()

        # Build UI
        self._build_controls()
        self._build_plots()

    def _build_controls(self):
        mf = ttk.Frame(self.root, padding=10)
        mf.pack(fill=tk.BOTH, expand=True)
        self.main_frame = mf

        # Row: directory + buttons + duration
        self.control_frame = ttk.LabelFrame(mf, text="Controls", padding=10)
        self.control_frame.pack(fill=tk.X)

        ttk.Label(self.control_frame, text="Data directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.dir_var = tk.StringVar(value=self.processor.data_dir)
        ttk.Entry(self.control_frame, textvariable=self.dir_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Browse…", command=self._browse).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Duration (s):").grid(row=0, column=3, sticky=tk.E, padx=5, pady=5)
        self.duration_var = tk.StringVar(value="0")  # 0 = run until Stop
        ttk.Entry(self.control_frame, textvariable=self.duration_var, width=8).grid(row=0, column=4, padx=5, pady=5)

        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_capture)
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.start_button.grid(row=0, column=5, padx=5, pady=5)
        self.stop_button.grid(row=0, column=6, padx=5, pady=5)

        # Status
        self.status_frame = ttk.LabelFrame(self.main_frame, text="Status", padding=10)
        self.status_frame.pack(fill=tk.X, pady=8)
        self.status_var = tk.StringVar(value="Ready")
        self.time_var = tk.StringVar(value="Time: 0.0 s")
        self.frames_var = tk.StringVar(value="Frames: 0")
        self.distance_var = tk.StringVar(value="Distance: 0.00 m")
        self.velocity_var = tk.StringVar(value="Velocity: 0.00 m/s")
        self.angle_var = tk.StringVar(value="Angle: 0.0°")
        ttk.Label(self.status_frame, textvariable=self.status_var).grid(row=0, column=0, padx=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.time_var).grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.frames_var).grid(row=0, column=2, padx=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.distance_var).grid(row=0, column=3, padx=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.velocity_var).grid(row=0, column=4, padx=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.angle_var).grid(row=0, column=5, padx=5, sticky=tk.W)
        
        # Per-antenna *unfiltered* distances (S1/S2/S3)
        self.s1_distance_var = tk.StringVar(value="S1 dist: -- m")
        self.s2_distance_var = tk.StringVar(value="S2 dist: -- m")
        self.s3_distance_var = tk.StringVar(value="S3 dist: -- m")

        ttk.Label(self.status_frame, textvariable=self.s1_distance_var).grid(row=1, column=3, padx=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.s2_distance_var).grid(row=1, column=4, padx=5, sticky=tk.W)
        ttk.Label(self.status_frame, textvariable=self.s3_distance_var).grid(row=1, column=5, padx=5, sticky=tk.W)


    def _build_plots(self):
        self.plots_frame = ttk.LabelFrame(self.main_frame, text="Time Series", padding=10)
        self.plots_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(11, 7), dpi=100)
        self.ax_dist = self.fig.add_subplot(311)
        self.ax_vel = self.fig.add_subplot(312)
        self.ax_ang = self.fig.add_subplot(313)

        self.ax_dist.set_ylabel("Distance (m)"); self.ax_vel.set_ylabel("Velocity (m/s)"); self.ax_ang.set_ylabel("Angle (°)")
        self.ax_ang.set_xlabel("Time (s)")
        self.dist_line, = self.ax_dist.plot([], [], label='Distance')
        self.vel_line, = self.ax_vel.plot([], [], label='Velocity')
        self.ang_line,  = self.ax_ang.plot([], [], label='Angle')

        self.ax_dist.legend(); self.ax_vel.legend(); self.ax_ang.legend()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(self.plots_frame); toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.fig.tight_layout()

    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.processor.data_dir)
        if d:
            self.processor.data_dir = d
            self.dir_var.set(d)

    def start_capture(self):
        try:
            dur = float(self.duration_var.get() or 0.0)
        except ValueError:
            dur = 0.0
        self.end_time = (time.time() + dur) if dur > 0 else None

        # reset processor buffers for a fresh trial
        self.processor.reset()

        # clear plot lines so previous trial disappears
        self.dist_line.set_data([], [])
        self.vel_line.set_data([], [])
        self.ang_line.set_data([], [])
        self.canvas.draw_idle()

        self.processing_active = True
        self.status_var.set("Capturing…")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

        self._tick()

    
    def stop_capture(self):
        if self.processing_active:
            self.processing_active = False

    def _tick(self):
        if not self.processing_active:
            self._save_and_cleanup()
            return

        if self.end_time and time.time() >= self.end_time:
            self.processing_active = False
            self._save_and_cleanup()
            return

        try:
            res = self.processor.process_frame()
            # Update status strings
            self.time_var.set(f"Time: {self.processor.timestamps[-1]:.1f} s")
            self.frames_var.set(f"Frames: {self.processor.frame_index}")
            self.distance_var.set(f"Distance: {res['distance']:.2f} m")
            self.velocity_var.set(f"Velocity: {res['velocity']:.2f} m/s")
            self.angle_var.set(f"Angle: {res['angle']:.1f}°")
            # Update plots
            ts = self.processor.timestamps
            self.dist_line.set_data(ts, self.processor.distances)
            self.vel_line.set_data(ts, self.processor.velocities)
            self.ang_line.set_data(ts, self.processor.angles)
            
            # Unfiltered per-sensor distances (first 3 RX antennas)
            if self.processor.s1_distances:
                self.s1_distance_var.set(f"S1 dist: {self.processor.s1_distances[-1]:.2f} m")
            else:
                self.s1_distance_var.set("S1 dist: -- m")

            if self.processor.s2_distances:
                self.s2_distance_var.set(f"S2 dist: {self.processor.s2_distances[-1]:.2f} m")
            else:
                self.s2_distance_var.set("S2 dist: -- m")

            if self.processor.s3_distances:
                self.s3_distance_var.set(f"S3 dist: {self.processor.s3_distances[-1]:.2f} m")
            else:
                self.s3_distance_var.set("S3 dist: -- m")

            for ax, arr in [(self.ax_dist, self.processor.distances),
                            (self.ax_vel, self.processor.velocities),
                            (self.ax_ang, self.processor.angles)]:
                if ts:
                    ax.set_xlim(ts[0], ts[-1] if ts[-1] > 1 else 1)
                if arr:
                    lo = min(arr); hi = max(arr)
                    if lo == hi: lo -= 1; hi += 1
                    ax.set_ylim(lo - 0.1*abs(lo if lo else 1), hi + 0.1*abs(hi if hi else 1))
            self.canvas.draw_idle()
        except ErrorFrameAcquisitionFailed:
            # soft retry
            pass
        except Exception as e:
            messagebox.showerror("Error", f"Processing error: {e}")
            self.processing_active = False
            self._save_and_cleanup()
            return

        # schedule next
        self.root.after(10, self._tick)

    def _save_and_cleanup(self):
        try:
            self.processor.save_data()
        finally:
            self.processor.close()
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.status_var.set("Saved & ready")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=None, help='Directory to save output Excel file')
    args, _ = parser.parse_known_args()

    root = tk.Tk()
    app = RadarGUI(root)
    if args.data_dir:
        app.processor.data_dir = args.data_dir
        app.dir_var.set(args.data_dir)
    root.mainloop()


if __name__ == '__main__':
    main()
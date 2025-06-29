from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def round_detector_minima(df, min_duration=1.0):
    activity_type = input("What type of data are you analyzing? (Walking / Turning / STS): ").strip().lower()

    if activity_type not in ["walking", "turning", "sts"]:
        print("❌ Invalid input. Please enter Walking, Turning, or STS.")
        return {}

    # Unified label for dict key and printing
    label = activity_type.upper() if activity_type == "sts" else activity_type.capitalize()
    events = {label: []}

    time = df['time(s)'].values
    velocity = df["Velocity Reconstructed"].values
    angle = df["Angle Reconstructed"].values
    distance = df["Distance Reconstructed"].values

    if activity_type == "walking":
        inv_velocity = -velocity
        minima_indices, _ = find_peaks(inv_velocity, distance=50)
        starts = minima_indices[:-1]
        stops = minima_indices[1:]
        for start, stop in zip(starts, stops):
            if time[stop] - time[start] >= min_duration:
                events[label].append((start, stop))

    elif activity_type == "turning":
        peak_indices, _ = find_peaks(np.abs(angle), distance=20, prominence=3)
        starts = peak_indices[:-1]
        stops = peak_indices[1:]
        for start, stop in zip(starts, stops):
            if time[stop] - time[start] >= min_duration:
                events[label].append((start, stop))

    elif activity_type == "sts":
      inv_distance = -distance
      minima_indices, _ = find_peaks(inv_distance, distance=20, prominence=0.2)
      starts = minima_indices[:-1]
      stops = minima_indices[1:]
      for start, stop in zip(starts, stops):
          if time[stop] - time[start] >= min_duration:
              events[label].append((start, stop))


    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(15, 5))

    if activity_type == "walking":
        ax.plot(time, velocity, label="Velocity (Reconstructed)")
        ax.set_ylabel("Velocity (m/s)")
    elif activity_type == "turning":
        ax.plot(time, angle, label="Angle (Reconstructed)")
        ax.set_ylabel("Angle (degrees)")
    else:
        ax.plot(time, distance, label="Distance (Reconstructed)")
        ax.set_ylabel("Distance (m)")

    for i, (start, stop) in enumerate(events[label], start=1):
        t_start = time[start]
        t_stop = time[stop]
        ax.axvline(t_start, color='k', linestyle='--', alpha=0.7)
        ax.axvline(t_stop, color='k', linestyle='--', alpha=0.7)
        ax.fill_betweenx(ax.get_ylim(), t_start, t_stop, color='gray', alpha=0.2)
        ax.text((t_start + t_stop)/2, ax.get_ylim()[1]*0.95, f"#{i}", ha='center', fontsize=8, color='red')

    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Print events
    print(f"\n📌 {label} Events:")
    for i, (start, stop) in enumerate(events[label], start=1):
        print(f"Round {i}: from index {start} to {stop}")

    return events

def main():
    # Load the Excel file saved by main()
    df = pd.read_excel("emdandeemd_output.xlsx")

    # 🔧 Rename columns to match what the round_detector_minima expects
    df = df.rename(columns={
        'Time (s)': 'time(s)',
        'Distance (m)': 'distance(m)',
        'Velocity (m/s)': 'velocity(m/s)',
        'Angle (degrees)': 'angle(degrees)'
    })


    # Run the round detector
    events = round_detector_minima(df)

    # Print detected events
    for label, spans in events.items():
        print(f"\n📌 {label} Events:")
        for i, (start, stop) in enumerate(spans):
            print(f"  Round {i+1}: from index {start} to {stop}")

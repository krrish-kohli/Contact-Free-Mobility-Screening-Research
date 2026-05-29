# Contact-Free Mobility Screening Research

This repository contains the Python-based implementation of a contact-free 60 GHz radar system designed to automatically measure performance for the Five-Times Sit-to-Stand (5×STS) test. This research aims to provide a private, non-intrusive alternative for assessing leg strength and balance in home settings.

## Overview
The 5×STS test is a critical clinical tool for assessing balance and fall risk. However, manual timing is often prone to variability, and traditional monitoring methods like wearables or cameras can be intrusive. This system utilizes a 60 GHz Infineon FMCW radar to detect sit-to-stand events without requiring markers or physical attachments.

## Key Features
* **Non-Intrusive:** Contact-free monitoring using radar technology.
* **Real-time Processing:** Latency of <20 ms per frame.
* **Clinical Validation:** Demonstrates clinically acceptable error rates compared to manual stop-watch timing.

## Technical Implementation
* **Hardware:** 60 GHz Infineon FMCW radar (3TX/4RX), tripod-mounted 1.5 m in front of the chair.
* **Software:** Python-based processing using NumPy and SciPy.
* **Data Processing Pipeline:**
    1. Radar data acquisition (128 chirps × 256 range samples @ 5 Hz).
    2. Median filter for noise reduction.
    3. Velocity thresholding ($\pm 0.15$ m/s) to identify stand/sit events.
    4. Automated duration calculation: $\Delta T = T_5 - T_0$.

## Results
In pilot testing with healthy adults, the radar system successfully performed 5×STS timing with a mean absolute error of $3.35 \pm 0.88$ (~16%) compared to reference stopwatch measurements.

## Future Directions
Future work aims to extract additional movement features such as ascent speed, sway, and fatigue, and to validate the system with older adults and stroke survivors in home environments.

## References
* Centers for Disease Control and Prevention (CDC). *Stroke Facts*. 2024.
* Centers for Disease Control and Prevention (CDC). *Important Facts About Falls*. 2024.
* National Council on Aging (NCOA). *Get the Facts on Falls Prevention*. 2023.
* Schmid, A. A., et al. (2012). *Prevalence and Predictors of Falls in People With Stroke: The LEAPS Study*.

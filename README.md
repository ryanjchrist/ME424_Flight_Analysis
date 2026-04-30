# ME 424 — Electric Glider Flight Performance Analysis

**Duke University, ME 424L — Spring 2026**  
Created by Ryan Christ

## Authorship and Validation

- Analysis and telemetry scripts in `src/` were authored with assistance from **Cursor & ChatGPT**.
- Ryan Christ tuned analysis parameters, curated the input logs, and verified all reported results and figures.

---

## Project Overview

This repository contains the flight data and analysis code for a modified RC glider project. The team designed and fabricated a trailing-edge flap mechanism for a Skynetic Bixler-style glider, then instrumented it with an ArduPilot flight controller, digital pitot tube, and DroneBridge ESP32-C3 telemetry link to collect high-frequency in-flight data.

The primary performance goals were a ≥10% reduction in stall speed and a ≥5% reduction in sink rate relative to the unmodified baseline glider. Three test configurations were evaluated:

- **Baseline** — stock glider, no flap mechanism (Log 24)
- **Flap Retracted** — modified wing installed, flap held up (Logs 119 + 128)
- **Flap Deployed** — flap fully deflected, increasing camber (Logs 112 + 119 deployed)

All flights were conducted at Blackwood Field, Duke Forest, with FAA UAS clearance filed for each session.

---

## Repository Structure

```
ME424_FlightAnalysis/
├── README.md
├── requirements.txt
├── src/
│   ├── glider_log_analysis.py      # Main post-flight analysis script
│   └── mavlink_realtime_plot.py    # Live telemetry dashboard (used during flight)
├── logs/
│   ├── log_24_UnknownDate.bin      # Baseline flight — Feb 27, 2026
│   ├── log_112_UnknownDate.bin     # Fixed flap deployed — Apr 8, 2026
│   ├── log_119_UnknownDate.bin     # Pilot-switched flap — Apr 8, 2026
│   └── log_128_UnknownDate.bin     # Flap retracted — Apr 24, 2026
└── figs/
    ├── fig_dash_log24.png           # Per-flight data dashboard: Log 24
    ├── fig_dash_log112.png          # Per-flight data dashboard: Log 112
    ├── fig_dash_log119.png          # Per-flight data dashboard: Log 119
    ├── fig_dash_log128.png          # Per-flight data dashboard: Log 128
    ├── fig_telemetry_overview.png   # All four logs side-by-side (5 channels each)
    ├── fig_combined_four.png        # All configs overlaid: polar, pitch, alt, sink
    └── fig5_group_polar.png         # Three-group glide polar comparison
```

---

## Setup

**Python 3.9+** is required.

```bash
pip install -r requirements.txt
```

Dependencies: `pymavlink`, `matplotlib`, `numpy`, `pyserial`

---

## Running the Analysis

The analysis script expects the `.bin` log files to be in the same directory as the script, or you can update the `log_files` paths in `main()`.

```bash
cd src
cp ../logs/*.bin .
python glider_log_analysis.py
```

Output figures are saved to the working directory. Move them to `figs/` after running if you want to keep the repo organized.

The live telemetry dashboard (`mavlink_realtime_plot.py`) is run during flight on a ground laptop connected to the DroneBridge ESP32-C3 via USB. It displays a real-time 2×2 panel of altitude, airspeed, glide polar, and airspeed distribution.

```bash
python mavlink_realtime_plot.py --port /dev/tty.usbserial-XXXX --baud 115200
```

---

## Output Figures

| File | Description |
|------|-------------|
| `fig_dash_log*.png` | 6-panel per-flight dashboard: altitude, airspeed, climb rate, attitude, throttle, flap. Glide segments highlighted in red. |
| `fig_telemetry_overview.png` | All four logs side-by-side with motor-on (orange) and flap-deployed (blue) shading. |
| `fig_combined_four.png` | All configurations overlaid: sink rate vs. airspeed (with quadratic trend), pitch vs. airspeed, altitude vs. time, sink rate vs. time. |
| `fig5_group_polar.png` | Three-group glide polar: binned median curves and IQR bands. Dashed horizontal lines mark each group's overall median sink rate. |

---

## Glide Segment Filtering

Samples must simultaneously satisfy all of the following to be included in the polar and performance analysis:

| Criterion | Value |
|-----------|-------|
| Throttle (C1 PWM) | ≤ 1200 µs (motor off) |
| Altitude AGL | ≥ 2 m |
| Corrected airspeed | ≥ 2 m/s |
| Roll angle | ≤ ±30° |
| Sink rate | ≥ 0.1 m/s |

---

## Stall Speed Methodology

Direct stall maneuvers were not performed — at RC glider scale an intentional stall risks structural damage and yields only a single data point. Instead, stall speed is estimated from the lowest airspeed readings recorded during normal unpowered glide.

During each glide the pilot modulated pitch continuously, producing a range of airspeeds from slow near-stall flight to moderate cruise. The lowest 10% of airspeed samples from each configuration's glide segments are averaged to produce the stall speed estimate. This provides multiple data points for redundancy against individual sensor noise or gust spikes, rather than relying on a single event. A lower value between configurations indicates the aircraft was routinely flying at slower controlled airspeeds, consistent with a higher C_L_max from flap deployment.

The 10% threshold was chosen empirically — it captured the slow-flight tail of the distribution cleanly across all three configurations without being pulled down by isolated outliers. The specific value is not claimed to be a formally validated standard; it is a practical engineering choice for this dataset.

---

## Airspeed Bias Correction

The pitot tube zero reading drifts slightly with temperature. A bias correction is computed from the median of idle-throttle airspeed samples in the 30 seconds immediately before first liftoff (the most contemporaneous zero-wind reference available). The corrected airspeed is `raw - bias` throughout all analysis.

| Log | Bias (m/s) | Pre-liftoff samples |
|-----|-----------|-------------------|
| 24  | 1.75 | ~30 |
| 112 | 1.17 | 2 (short ground period — use with caution) |
| 119 | ~0.9 | 157 |
| 128 | ~0.8 | 184 |

---

## Avionics

| Component | Role |
|-----------|------|
| Matek F405-Wing V2 | ArduPilot flight controller, DataFlash logger |
| MS4525DO digital pitot | Airspeed measurement via I²C |
| SIM28 GPS module | Position and groundspeed (not used for airspeed) |
| ESP32-C3 DroneBridge | ESP-NOW LR telemetry to ground laptop |
| SG90 servos (×4) | Flap and aileron actuation |
| ArduPlane FBWA mode | Roll/pitch auto-stabilization during flight |

---

## Notes

- Logs 110, 111, 115 are ground tests only and are not included here.
- Log 119 is capped at 240 seconds in the analysis — the aircraft was on the ground after that point despite the log continuing.
- The barometer logs at ~0.7 Hz in logs 24 and 112, and ~8.5 Hz in logs 119 and 128. Sample counts per flight are small; interpret trends accordingly.

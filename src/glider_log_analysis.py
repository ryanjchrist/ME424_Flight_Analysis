#!/usr/bin/env python3
"""
ME 424 – Electric glider performance analysis
==============================================
Flight logs
-----------
  log_24   Baseline flight – no flap, no aileron extension
  log_110  Ground test only (~18 s, never airborne)
  log_111  Ground test only (~20 s, never airborne)
  log_112  Fixed-flap flight – RCOU C5 held at 1100 throughout (~7 min)
  log_115  Ground test / servo sweep (altitude ≈ 0 throughout)
  log_119  Switch-controlled flap flight (~12 min);
             RCOU C5: 1500 = flap retracted, 1900 = flap deployed
  log_128  Flap-retracted flight (~9 min); RCOU C5 ≈ 1500 throughout

Servo mapping (ArduPilot RCOU channel → physical servo)
  C1 = throttle   C2 = rudder   C3 = elevator
  C4 = aileron    C5 = flap

Airspeed bias correction
  The flight controller logs from power-on, so the ground period before
  liftoff can be 40+ minutes before actual flight (as in log_24).  The bias
  is computed from a 30-second window ending just before the first liftoff
  (first moment altitude exceeds 3 m AGL) where throttle is at idle.  This
  gives the most contemporaneous zero-wind reference for the pitot tube.

Glide-segment filter (all must be true simultaneously)
  throttle C1 ≤ 1200   motor off / prop windmilling excluded
  altitude AGL ≥ 2 m   above ground effect / taxi noise
  corrected airspeed ≥ 2 m/s   removes sensor noise at rest
  |roll| ≤ 30°          wings-level (uncoordinated turns affect polar)
  sink rate ≥ 0.1 m/s   actually descending

Overview plots are cropped to the detected flight window (±30 s margin)
so the brief airborne periods are not buried in hours of ground logging.

NOTE ON SAMPLE SIZE
  The barometer logs at ≈ 0.7 Hz in log_24 and log_112, and ≈ 8.5 Hz
  in log_119.  Each flight is 50–140 s long, so sample counts are small;
  interpret trends with appropriate caution.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from pymavlink.DFReader import DFReader_binary
except ImportError:
    print("ERROR: pymavlink not installed. Run:  pip install pymavlink", file=sys.stderr)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

THROTTLE_IDLE      = 1200   # C1 PWM ≤ this → motor off
ALT_MIN_M          = 2.0    # minimum AGL for glide samples
MIN_AIRSPEED_MS    = 2.0    # minimum corrected airspeed
MAX_ROLL_DEG       = 30.0   # wings-level threshold
SINK_MIN_MS        = 0.10   # must be actually descending
FLAP_UP_MAX_PWM    = 1600   # C5 ≤ this → flap retracted
FLAP_DOWN_MIN_PWM  = 1700   # C5 ≥ this → flap deployed
BIAS_LOOKBACK_S    = 30     # seconds before first liftoff to use for bias
LIFTOFF_ALT_M      = 3.0    # altitude that marks liftoff for bias window
FLIGHT_ALT_M       = 3.0    # altitude threshold used to detect airborne window
PLOT_MARGIN_S      = 5      # seconds of ground data to show either side of flight

STALL_REDUCTION_GOAL = 0.10   # 10 % stall-speed reduction target
SINK_REDUCTION_GOAL  = 0.05   # 5 % sink-rate reduction target

# Individual dataset identifiers (per-log detail plots)
DATASETS = {
    "baseline":    dict(label="Baseline – no flap (log 24)",         color="#2166ac"),
    "fixed_flap":  dict(label="Fixed flap – C5=1100 (log 112)",      color="#e08214"),
    "flap_up":     dict(label="Flap retracted – C5≈1500 (log 119)",  color="#4dac26"),
    "flap_down":   dict(label="Flap deployed  – C5≈1900 (log 119)",  color="#d01c8b"),
    "flap_up_128": dict(label="Flap retracted – C5≈1500 (log 128)",  color="#35978f"),
    # ── Group-level entries (comparison figures) ──────────────────────────────
    # Group A: original unmodified glider
    "group_baseline": dict(
        label="A – Baseline  (original wing, log 24)",
        color="#2166ac",
    ),
    # Group B: modified wing with flap mechanism installed but flap retracted
    "group_extended": dict(
        label="B – Extended wing, flap retracted  (logs 119 + 128)",
        color="#1b7837",
    ),
    # Group C: flap deployed or held at a fixed deployed angle
    "group_deployed": dict(
        label="C – Flap deployed / fixed  (log 112 fixed + log 119 down)",
        color="#762a83",
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# Log parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_log(path: Path) -> dict:
    """Read a DataFlash .bin log and return raw numpy arrays."""
    log = DFReader_binary(str(path))

    baro_t, baro_alt, baro_crt = [], [], []
    arsp_t, arsp_v             = [], []
    att_t,  roll,    pitch     = [], [], []
    rcou_t = []
    rcou   = {f"C{i}": [] for i in range(1, 15)}
    rcin_t = []
    rcin   = {f"C{i}": [] for i in range(1, 15)}

    while True:
        m = log.recv_msg()
        if m is None:
            break
        ts = getattr(m, "TimeUS", None)
        if ts is not None:
            ts /= 1e6
        else:
            ts = getattr(m, "TimeMS", None)
            if ts is not None:
                ts /= 1e3
        if ts is None:
            ts = getattr(m, "_timestamp", None)
        if ts is None:
            continue

        typ = m.get_type()
        if typ == "BARO":
            baro_t.append(ts)
            baro_alt.append(getattr(m, "Alt", np.nan))
            baro_crt.append(getattr(m, "CRt", np.nan))
        elif typ == "ARSP":
            arsp_t.append(ts)
            arsp_v.append(getattr(m, "Airspeed", np.nan))
        elif typ == "ATT":
            att_t.append(ts)
            roll.append(getattr(m, "Roll", np.nan))
            pitch.append(getattr(m, "Pitch", np.nan))
        elif typ == "RCOU":
            rcou_t.append(ts)
            for k in rcou:
                rcou[k].append(getattr(m, k, np.nan))
        elif typ == "RCIN":
            rcin_t.append(ts)
            for k in rcin:
                rcin[k].append(getattr(m, k, np.nan))

    log.close()

    def arr(x):
        return np.asarray(x, dtype=float)

    return dict(
        baro_t=arr(baro_t), baro_alt=arr(baro_alt), baro_crt=arr(baro_crt),
        arsp_t=arr(arsp_t), arsp_v=arr(arsp_v),
        att_t=arr(att_t), roll=arr(roll), pitch=arr(pitch),
        rcou_t=arr(rcou_t), rcou={k: arr(v) for k, v in rcou.items()},
        rcin_t=arr(rcin_t), rcin={k: arr(v) for k, v in rcin.items()},
    )


def _interp(src_t, src_y, dst_t):
    src_t = np.asarray(src_t, dtype=float)
    src_y = np.asarray(src_y, dtype=float)
    dst_t = np.asarray(dst_t, dtype=float)
    ok = np.isfinite(src_t) & np.isfinite(src_y)
    if ok.sum() < 2 or dst_t.size == 0:
        return np.full(dst_t.shape, np.nan)
    idx = np.argsort(src_t[ok])
    return np.interp(dst_t, src_t[ok][idx], src_y[ok][idx], left=np.nan, right=np.nan)


# ──────────────────────────────────────────────────────────────────────────────
# Build analysis frame on the BARO time grid
# ──────────────────────────────────────────────────────────────────────────────

def build_frame(raw: dict) -> dict:
    t = raw["baro_t"]
    if t.size == 0:
        raise ValueError("No BARO data found in log.")

    # ── Altitude AGL reference ────────────────────────────────────────────────
    # Use a window of samples clearly on the ground (first samples where
    # barometer is stable near its initial value).
    n_ref = min(max(20, int(0.03 * t.size)), 200)
    ground_baro = np.nanmedian(raw["baro_alt"][:n_ref])
    alt_agl = raw["baro_alt"] - ground_baro
    sink_ms = -raw["baro_crt"]

    # ── Interpolate all streams onto BARO time grid ───────────────────────────
    airspeed_raw = _interp(raw["arsp_t"], raw["arsp_v"], t)
    roll_deg     = _interp(raw["att_t"],  raw["roll"],   t)
    pitch_deg    = _interp(raw["att_t"],  raw["pitch"],  t)
    throttle_pwm = _interp(raw["rcou_t"], raw["rcou"]["C1"], t)
    flap_pwm     = _interp(raw["rcou_t"], raw["rcou"]["C5"], t)

    # ── Detect flight window ──────────────────────────────────────────────────
    # "In the air" requires altitude above threshold AND either throttle active
    # or corrected airspeed above threshold.  This avoids treating the glider
    # sitting on elevated terrain (constant low airspeed, idle throttle) as
    # still being airborne.
    # bias is not yet known here, so use raw airspeed as a rough proxy;
    # the bias is typically small relative to flight speeds
    in_air_mask = (
        (alt_agl > FLIGHT_ALT_M)
        & (
            (np.isfinite(throttle_pwm) & (throttle_pwm > THROTTLE_IDLE))
            | (np.isfinite(airspeed_raw) & (airspeed_raw >= MIN_AIRSPEED_MS))
        )
    )
    # Fallback to pure altitude if no combined samples found
    if not in_air_mask.any():
        in_air_mask = alt_agl > FLIGHT_ALT_M

    liftoff_idx = int(np.argmax(in_air_mask)) if in_air_mask.any() else len(t) - 1
    liftoff_t   = float(t[liftoff_idx])
    landing_idx = int(len(t) - 1 - np.argmax(in_air_mask[::-1])) if in_air_mask.any() else 0
    landing_t   = float(t[landing_idx])

    # ── Refine landing using attitude activity ────────────────────────────────
    # After the actual landing, roll and pitch go flat (glider sitting still).
    # Find the last moment post-liftoff where attitude was still actively
    # changing, and use that if it is earlier than the altitude-based estimate.
    if in_air_mask.any() and roll_deg.size > 20:
        WIN = max(5, int(roll_deg.size * 0.01))   # ~1 % of log, min 5 samples
        roll_std  = np.array([np.nanstd(roll_deg [max(0, i-WIN): i+WIN+1])
                               for i in range(len(t))])
        pitch_std = np.array([np.nanstd(pitch_deg[max(0, i-WIN): i+WIN+1])
                               for i in range(len(t))])
        att_active = (roll_std + pitch_std) > 1.5   # degrees combined activity
        post_liftoff = t >= liftoff_t
        att_flight   = att_active & post_liftoff
        if att_flight.any():
            att_land_idx = int(len(t) - 1 - np.argmax(att_flight[::-1]))
            att_land_t   = float(t[att_land_idx])
            if att_land_t < landing_t:
                landing_t   = att_land_t
                landing_idx = att_land_idx

    # ── Airspeed bias correction ──────────────────────────────────────────────
    # Use the 30 s of idle-throttle data immediately before first liftoff.
    # This is the most contemporaneous zero-wind reference, regardless of how
    # long the log was running before the flight.
    bias_window = (
        (t >= liftoff_t - BIAS_LOOKBACK_S)
        & (t <  liftoff_t)
        & np.isfinite(airspeed_raw)
        & np.isfinite(throttle_pwm)
        & (throttle_pwm <= THROTTLE_IDLE)
        & (alt_agl < LIFTOFF_ALT_M)
    )
    if bias_window.sum() >= 3:
        bias = float(np.nanmedian(airspeed_raw[bias_window]))
    else:
        # Fallback: any ground samples (alt < 2 m, throttle idle) in full log
        ground_any = (
            np.isfinite(airspeed_raw) & np.isfinite(throttle_pwm)
            & (throttle_pwm <= THROTTLE_IDLE) & (alt_agl < 2.0)
        )
        bias = float(np.nanmedian(airspeed_raw[ground_any])) if ground_any.sum() >= 3 else 0.0
    airspeed = airspeed_raw - bias

    # ── Flap state ────────────────────────────────────────────────────────────
    # -1 = unknown  0 = retracted  1 = deployed  (transitioning stays -1)
    flap_state = np.full(t.shape, -1, dtype=int)
    has_flap = np.isfinite(flap_pwm) & (flap_pwm > 900)
    flap_state[has_flap & (flap_pwm <= FLAP_UP_MAX_PWM)]  = 0
    flap_state[has_flap & (flap_pwm >= FLAP_DOWN_MIN_PWM)] = 1

    # ── Glide-segment mask ────────────────────────────────────────────────────
    is_glide = (
        np.isfinite(alt_agl)        & (alt_agl >= ALT_MIN_M)
        & np.isfinite(airspeed)     & (airspeed >= MIN_AIRSPEED_MS)
        & np.isfinite(sink_ms)      & (sink_ms  >= SINK_MIN_MS)
        & np.isfinite(throttle_pwm) & (throttle_pwm <= THROTTLE_IDLE)
        & np.isfinite(roll_deg)     & (np.abs(roll_deg) <= MAX_ROLL_DEG)
    )

    return dict(
        t=t,
        # Time relative to liftoff (negative = before flight, positive = after)
        rel_t=t - liftoff_t,
        alt_agl=alt_agl, sink_ms=sink_ms,
        airspeed_raw=airspeed_raw, airspeed=airspeed, airspeed_bias=bias,
        roll_deg=roll_deg, pitch_deg=pitch_deg,
        throttle_pwm=throttle_pwm, flap_pwm=flap_pwm, flap_state=flap_state,
        is_glide=is_glide,
        bias_samples=int(bias_window.sum()),
        liftoff_t=liftoff_t, landing_t=landing_t,
        duration_s=float(landing_t - liftoff_t),
        max_alt_m=float(np.nanmax(alt_agl)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Binned glide polar
# ──────────────────────────────────────────────────────────────────────────────

def bin_polar(speeds: np.ndarray, sinks: np.ndarray, bin_width: float = 1.5):
    """Return (centers, medians, q25, q75); bins with < 3 points are skipped."""
    lo = np.floor(np.nanmin(speeds))
    hi = np.ceil(np.nanmax(speeds)) + bin_width
    edges = np.arange(lo, hi, bin_width)
    centers, medians, q25_arr, q75_arr = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (speeds >= a) & (speeds < b)
        if sel.sum() < 3:
            continue
        centers.append(0.5 * (a + b))
        medians.append(float(np.nanmedian(sinks[sel])))
        q25_arr.append(float(np.nanpercentile(sinks[sel], 25)))
        q75_arr.append(float(np.nanpercentile(sinks[sel], 75)))
    return (np.array(centers), np.array(medians),
            np.array(q25_arr),  np.array(q75_arr))


# ──────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ──────────────────────────────────────────────────────────────────────────────

def _shade_flap_deployed(axes, t_arr, flap_state_arr, color="#d01c8b", alpha=0.08):
    deployed = flap_state_arr == 1
    if not deployed.any():
        return
    starts = np.where(deployed & ~np.r_[False, deployed[:-1]])[0]
    ends   = np.where(deployed & ~np.r_[deployed[1:], False])[0]
    for s, e in zip(starts, ends):
        for ax in axes:
            ax.axvspan(t_arr[s], t_arr[e], alpha=alpha, color=color, zorder=0)


# ──────────────────────────────────────────────────────────────────────────────
# Glide polar (all datasets)
# ──────────────────────────────────────────────────────────────────────────────

def plot_polar(datasets: dict, out_path: Path):
    """datasets: key → (speeds array, sinks array)

    Draws the binned-median polar curve for each dataset.  A dashed horizontal
    line marks the overall median sink rate — a more robust metric than the
    polar minimum when sample sizes differ across groups.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Glide polar – throttle-off, wings-level segments\n"
                 "(shaded band = IQR; dashed line = group median sink rate)",
                 fontsize=11, fontweight="bold")

    median_sinks = {}
    for key, (spd, snk) in datasets.items():
        if spd.size < 3:
            continue
        cfg   = DATASETS[key]
        color = cfg["color"]
        label = cfg["label"]

        ax.scatter(spd, snk, s=14, alpha=0.25, color=color, zorder=2)

        cx, med, q25, q75 = bin_polar(spd, snk)
        med_sink = float(np.nanmedian(snk))
        if cx.size >= 2:
            ax.plot(cx, med, lw=2.5, color=color,
                    label=f"{label} (n={spd.size}, med sink={med_sink:.2f} m/s)", zorder=4)
            ax.fill_between(cx, q25, q75, color=color, alpha=0.15, zorder=1)
            ax.axhline(med_sink, color=color, lw=1.2, ls="--", alpha=0.7, zorder=3)
        else:
            ax.plot([], [], color=color,
                    label=f"{label} (n={spd.size}, med sink={med_sink:.2f} m/s, sparse)")
            ax.axhline(med_sink, color=color, lw=1.2, ls="--", alpha=0.7, zorder=3)
        median_sinks[key] = med_sink

    ax.set_xlabel("Corrected airspeed (m/s)", fontsize=11)
    ax.set_ylabel("Sink rate (m/s)  [positive = descending]", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return median_sinks   # key → median sink rate (m/s)




# ──────────────────────────────────────────────────────────────────────────────
# Per-flight data dashboard (single log, all channels)
# ──────────────────────────────────────────────────────────────────────────────

def plot_avionics_dashboard(frame: dict, log_label: str, out_path: Path):
    """
    6-panel flight data summary cropped to the airborne window.
    Panels (top to bottom):
      1. Altitude AGL
      2. Airspeed (raw faint, bias-corrected bold; glide segments marked)
      3. Climb/sink rate  (+ve = climbing)
      4. Roll & Pitch angles
      5. Throttle (%)
      6. Flap (%) — omitted for logs without a flap servo
    """
    # ── Crop to airborne window ───────────────────────────────────────────────
    flight_dur = frame["landing_t"] - frame["liftoff_t"]
    t_lo = -PLOT_MARGIN_S
    t_hi = flight_dur + PLOT_MARGIN_S
    crop = (frame["rel_t"] >= t_lo) & (frame["rel_t"] <= t_hi)

    t   = frame["rel_t"][crop]
    alt = frame["alt_agl"][crop]
    arsp_raw = frame["airspeed_raw"][crop]
    arsp     = frame["airspeed"][crop]
    climb    = -frame["sink_ms"][crop]          # positive = climbing (normal convention)
    roll     = frame["roll_deg"][crop]
    pitch    = frame["pitch_deg"][crop]
    thr      = frame["throttle_pwm"][crop]
    flap     = frame["flap_pwm"][crop]
    fs       = frame["flap_state"][crop]
    gm       = frame["is_glide"][crop]

    has_flap = np.isfinite(flap).any() and (flap[np.isfinite(flap)] > 900).any()
    nrows = 6 if has_flap else 5

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        10,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.linewidth":   0.8,
        "grid.linewidth":   0.5,
        "xtick.direction":  "in",
        "ytick.direction":  "in",
    })

    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(11, 1.75 * nrows),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    fig.suptitle(
        f"Flight data – {log_label}   |   ArduPilot DataFlash log",
        fontsize=11, fontweight="bold", y=0.995,
    )

    # Shade motor-on periods across all panels
    motor_on = np.isfinite(thr) & (thr > THROTTLE_IDLE)
    if motor_on.any():
        starts = np.where(motor_on & ~np.r_[False, motor_on[:-1]])[0]
        ends   = np.where(motor_on & ~np.r_[motor_on[1:], False])[0]
        for s, e in zip(starts, ends):
            for ax in axes:
                ax.axvspan(t[s], t[e], color="#f4a582", alpha=0.22, zorder=0)

    # Shade flap-deployed periods across all panels
    if has_flap:
        _shade_flap_deployed(axes, t, fs, color="#92c5de", alpha=0.22)

    # ── Panel 0 – Altitude AGL ────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t, 0, alt, alpha=0.18, color="#2166ac")
    ax.plot(t, alt, color="#2166ac", lw=1.4)
    ax.set_ylabel("Alt AGL\n(m)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_label_coords(-0.07, 0.5)

    # ── Panel 1 – Airspeed ────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, arsp_raw, color="#d6604d", lw=0.6, alpha=0.35,
            label=f"Raw (−{frame['airspeed_bias']:.1f} m/s bias)")
    ax.plot(t, arsp,     color="#d6604d", lw=1.4, label="Corrected")
    if gm.any():
        ax.scatter(t[gm], arsp[gm], s=18, color="#b2182b", zorder=4,
                   edgecolors="white", linewidths=0.4, label="Glide segment")
    ax.set_ylabel("Airspeed\n(m/s)", fontsize=9)
    ax.yaxis.set_label_coords(-0.07, 0.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.7,
              ncol=3, handlelength=1.2, handletextpad=0.4)

    # ── Panel 2 – Climb rate (positive = up) ─────────────────────────────────
    ax = axes[2]
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.fill_between(t, 0, climb, where=(climb > 0), alpha=0.25, color="#4dac26",
                    interpolate=True, label="Climbing")
    ax.fill_between(t, 0, climb, where=(climb < 0), alpha=0.25, color="#d01c8b",
                    interpolate=True, label="Descending")
    ax.plot(t, climb, color="0.3", lw=1.0)
    ax.set_ylabel("Climb rate\n(m/s)", fontsize=9)
    ax.yaxis.set_label_coords(-0.07, 0.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.7,
              ncol=2, handlelength=1.0, handletextpad=0.4)

    # ── Panel 3 – Roll & Pitch ────────────────────────────────────────────────
    ax = axes[3]
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.plot(t, roll,  color="#762a83", lw=1.2, label="Roll")
    ax.plot(t, pitch, color="#1b7837", lw=1.2, label="Pitch", alpha=0.85)
    ax.set_ylabel("Attitude\n(deg)", fontsize=9)
    ax.yaxis.set_label_coords(-0.07, 0.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.7,
              ncol=2, handlelength=1.0, handletextpad=0.4)

    # ── Panel 4 – Throttle ────────────────────────────────────────────────────
    ax = axes[4]
    # Normalise to 0–100 %  (1100 = 0 %, 1900 = 100 %)
    thr_pct = np.clip((thr - 1100) / 8.0, 0, 100)
    ax.fill_between(t, 0, thr_pct, alpha=0.35, color="#f4a582")
    ax.plot(t, thr_pct, color="#ca4c07", lw=1.2)
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.set_ylabel("Throttle\n(%)", fontsize=9)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_label_coords(-0.07, 0.5)

    # ── Panel 5 – Flap servo (optional) ──────────────────────────────────────
    if has_flap:
        ax = axes[5]
        # Map PWM to 0–100 % deployment  (1500 = 0 %, 1900 = 100 %)
        flap_pct = np.clip((flap - 1500) / 4.0, 0, 100)
        ax.fill_between(t, 0, flap_pct, alpha=0.35, color="#92c5de")
        ax.plot(t, flap_pct, color="#2166ac", lw=1.2)
        ax.axhline(0, color="0.5", lw=0.8, ls=":")
        ax.set_ylabel("Flap\n(%)", fontsize=9)
        ax.set_ylim(-5, 105)
        ax.yaxis.set_label_coords(-0.07, 0.5)

    axes[-1].set_xlabel("Time from first liftoff (s)", fontsize=9)

    # Shared legend for shading (motor / flap)
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor="#f4a582", alpha=0.55, label="Motor on")]
    if has_flap:
        legend_patches.append(Patch(facecolor="#92c5de", alpha=0.55, label="Flap deployed"))
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=8.5,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.01))

    # Tighten and save
    fig.tight_layout(rect=[0, 0.03, 1, 0.998])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Reset rcParams so other plots aren't affected
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Saved {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 10 – Four combined plots, all logs overlaid
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_four(frames: dict, out_path: Path):
    """
    2×2 figure with every log overlaid on the same axes:
      TL: Sink rate vs airspeed  (glide segments, with quadratic trend)
      TR: Pitch angle vs airspeed (glide segments, with linear trend)
      BL: Altitude AGL vs time   (full airborne window)
      BR: Sink rate vs time       (full airborne window)
    Log 119 is split into flap-up and flap-down series for TL/TR.
    """
    from numpy.polynomial.polynomial import polyfit, polyval

    SERIES_GLIDE = [
        dict(key="24",  label="Baseline – no flap (log 24)",
             color=DATASETS["baseline"]["color"],  flap_filter=None),
        dict(key="112", label="Fixed flap (log 112)",
             color=DATASETS["fixed_flap"]["color"], flap_filter=None),
        dict(key="119", label="Flap retracted (log 119)",
             color=DATASETS["flap_up"]["color"],    flap_filter=0),
        dict(key="119", label="Flap deployed (log 119)",
             color=DATASETS["flap_down"]["color"],  flap_filter=1),
        dict(key="128", label="Flap retracted (log 128)",
             color=DATASETS["flap_up_128"]["color"], flap_filter=None),
    ]
    SERIES_GLIDE = [s for s in SERIES_GLIDE if s["key"] in frames]

    SERIES_TIME = [
        dict(key="24",  label="Baseline – no flap (log 24)",
             color=DATASETS["baseline"]["color"]),
        dict(key="112", label="Fixed flap (log 112)",
             color=DATASETS["fixed_flap"]["color"]),
        dict(key="119", label="Flap switch (log 119)",
             color=DATASETS["flap_up"]["color"]),
        dict(key="128", label="Flap retracted (log 128)",
             color=DATASETS["flap_up_128"]["color"]),
    ]
    SERIES_TIME = [s for s in SERIES_TIME if s["key"] in frames]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Flight Data – All Logs", fontsize=13, fontweight="bold")

    ax_ss  = axes[0, 0]   # sink vs airspeed
    ax_ps  = axes[0, 1]   # pitch vs airspeed
    ax_at  = axes[1, 0]   # altitude vs time
    ax_st  = axes[1, 1]   # sink vs time

    # ── Glide-segment scatter plots ───────────────────────────────────────────
    for s in SERIES_GLIDE:
        fr    = frames[s["key"]]
        color = s["color"]
        label = s["label"]

        gm = fr["is_glide"].copy()
        if s["flap_filter"] is not None:
            gm = gm & (fr["flap_state"] == s["flap_filter"])

        ok_ss = gm & np.isfinite(fr["airspeed"]) & np.isfinite(fr["sink_ms"])
        ok_ps = gm & np.isfinite(fr["airspeed"]) & np.isfinite(fr["pitch_deg"])

        spd  = fr["airspeed"][ok_ss]
        sink = fr["sink_ms"][ok_ss]
        spd_p = fr["airspeed"][ok_ps]
        pitch = fr["pitch_deg"][ok_ps]

        if spd.size >= 2:
            ax_ss.scatter(spd, sink, s=22, alpha=0.6, color=color,
                          edgecolors="none", label=label, zorder=3)
            if spd.size >= 4:
                xs = np.linspace(spd.min(), spd.max(), 120)
                c  = polyfit(spd, sink, 2)
                ax_ss.plot(xs, polyval(xs, c), color=color, lw=2.0, alpha=0.9, zorder=4)

        if spd_p.size >= 2:
            ax_ps.scatter(spd_p, pitch, s=22, alpha=0.6, color=color,
                          edgecolors="none", label=label, zorder=3)
            if spd_p.size >= 4:
                xs2 = np.linspace(spd_p.min(), spd_p.max(), 80)
                c2  = polyfit(spd_p, pitch, 1)
                ax_ps.plot(xs2, polyval(xs2, c2), color=color, lw=2.0, alpha=0.9, zorder=4)

    # ── Time-series plots ─────────────────────────────────────────────────────
    for s in SERIES_TIME:
        fr    = frames[s["key"]]
        color = s["color"]
        label = s["label"]
        dur   = fr["duration_s"]

        crop = (fr["rel_t"] >= 0) & (fr["rel_t"] <= dur)
        t    = fr["rel_t"][crop]
        alt  = fr["alt_agl"][crop]
        sink = fr["sink_ms"][crop]

        ax_at.plot(t, alt,  color=color, lw=1.8, label=label, alpha=0.85)
        ax_st.plot(t, sink, color=color, lw=1.6, label=label, alpha=0.80)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax_ss.set_xlabel("Corrected airspeed (m/s)")
    ax_ss.set_ylabel("Sink rate (m/s)")
    ax_ss.set_title("Sink Rate vs Airspeed", fontweight="bold")
    ax_ss.legend(fontsize=8, loc="upper right")
    ax_ss.grid(True, alpha=0.3)

    ax_ps.set_xlabel("Corrected airspeed (m/s)")
    ax_ps.set_ylabel("Pitch angle (°)")
    ax_ps.set_title("Pitch vs Airspeed", fontweight="bold")
    ax_ps.axhline(0, color="0.5", lw=0.8, ls=":")
    ax_ps.legend(fontsize=8, loc="upper right")
    ax_ps.grid(True, alpha=0.3)

    ax_at.set_xlabel("Time from liftoff (s)")
    ax_at.set_ylabel("Altitude AGL (m)")
    ax_at.set_title("Altitude vs Time", fontweight="bold")
    ax_at.set_ylim(bottom=0)
    ax_at.legend(fontsize=8, loc="upper right")
    ax_at.grid(True, alpha=0.3)

    ax_st.set_xlabel("Time from liftoff (s)")
    ax_st.set_ylabel("Sink rate (m/s)")
    ax_st.set_title("Sink Rate vs Time", fontweight="bold")
    ax_st.axhline(0, color="0.5", lw=0.8, ls=":")
    ax_st.legend(fontsize=8, loc="upper right")
    ax_st.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# All-flights telemetry comparison — one column per flight, one row per channel
# ──────────────────────────────────────────────────────────────────────────────

def plot_telemetry_overview(frames: dict, out_path: Path):
    """
    Multi-column figure with one column per flight and one row per sensor channel.

    Columns  (left → right):  Log 24 | Log 112 | Log 119 | Log 128
    Rows     (top → bottom):
        0  Altitude AGL (m)
        1  Airspeed corrected (m/s)  — glide segments highlighted
        2  Roll & pitch (°)
        3  Throttle 0–100 %
        4  Flap 0–100 %   (hidden for log 24 which has no flap)

    Each column is cropped to its airborne window.
    Motor-on periods are shaded salmon; flap-deployed periods shaded blue.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          8.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.9,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.35,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
    })

    # flight-specific config
    cols = [
        dict(key="24",  title="Log 24 — Baseline\n(no flap)",              has_flap=False,
             col_color="#2166ac", max_dur=None),
        dict(key="112", title="Log 112 — Fixed Flap\n(C5 = 1100 PWM)",     has_flap=True,
             col_color="#e08214", max_dur=None),
        dict(key="119", title="Log 119 — Flap Switch\n(pilot-controlled)",  has_flap=True,
             col_color="#4dac26", max_dur=240),
        dict(key="128", title="Log 128 — Flap Retracted\n(C5 ≈ 1500 PWM)", has_flap=True,
             col_color="#35978f", max_dur=None),
    ]
    cols = [c for c in cols if c["key"] in frames]
    ncols = len(cols)

    row_labels = [
        "Altitude AGL\n(m)",
        "Airspeed\n(m/s)",
        "Roll / Pitch\n(°)",
        "Throttle\n(%)",
        "Flap\n(%)",
    ]
    NROWS = 5   # we always reserve space; hide flap row for no-flap columns

    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor("white")

    # GridSpec: 5 data rows × ncols, with extra space at top for column titles
    gs = gridspec.GridSpec(
        NROWS, ncols,
        figure=fig,
        top=0.88, bottom=0.09,
        left=0.07, right=0.98,
        hspace=0.30, wspace=0.30,
        height_ratios=[1.1, 1.0, 1.0, 0.8, 0.8],
    )

    axes_grid = [[fig.add_subplot(gs[row, col]) for col in range(ncols)]
                 for row in range(NROWS)]

    # ── Column loop ───────────────────────────────────────────────────────────
    for ci, cfg in enumerate(cols):
        frame     = frames[cfg["key"]]
        has_flap  = cfg["has_flap"]
        cc        = cfg["col_color"]   # accent color for this flight

        # Strict airborne crop — zero margin for slides
        liftoff_t  = frame["liftoff_t"]
        landing_t  = frame["landing_t"]
        dur        = landing_t - liftoff_t
        if cfg["max_dur"] is not None:
            dur = min(dur, cfg["max_dur"])
        crop       = (frame["rel_t"] >= 0) & (frame["rel_t"] <= dur)

        t       = frame["rel_t"][crop]
        alt     = frame["alt_agl"][crop]
        arsp    = frame["airspeed"][crop]
        roll    = frame["roll_deg"][crop]
        pitch   = frame["pitch_deg"][crop]
        thr     = frame["throttle_pwm"][crop]
        flap    = frame["flap_pwm"][crop]
        fs      = frame["flap_state"][crop]
        gm      = frame["is_glide"][crop]

        # Throttle → 0-100 %  (1100 = 0 %, 1900 = 100 %)
        thr_pct = np.clip((thr - 1100) / 8.0, 0, 100)
        # Flap    → 0-100 %  (1500 = 0 %, 1900 = 100 %)
        flap_pct = np.clip((flap - 1500) / 4.0, 0, 100) if has_flap else np.zeros_like(t)

        # Motor-on mask → shade across all 5 rows
        motor_on = np.isfinite(thr) & (thr > THROTTLE_IDLE)
        mo_starts = np.where(motor_on & ~np.r_[False, motor_on[:-1]])[0]
        mo_ends   = np.where(motor_on & ~np.r_[motor_on[1:], False])[0]

        # Flap-deployed → shade rows 0-4
        flap_dep   = (fs == 1)
        fd_starts  = np.where(flap_dep & ~np.r_[False, flap_dep[:-1]])[0]
        fd_ends    = np.where(flap_dep & ~np.r_[flap_dep[1:], False])[0]

        for row in range(NROWS):
            ax = axes_grid[row][ci]
            ax.set_xlim(0, dur)
            ax.grid(True, axis="y")

            # shade motor-on
            for s, e in zip(mo_starts, mo_ends):
                ax.axvspan(t[s], t[e], color="#f4a582", alpha=0.30, zorder=0, lw=0)

            # shade flap-deployed
            if has_flap:
                for s, e in zip(fd_starts, fd_ends):
                    ax.axvspan(t[s], t[e], color="#b2d8f7", alpha=0.38, zorder=0, lw=0)

        # ── Row 0: Altitude ───────────────────────────────────────────────────
        ax = axes_grid[0][ci]
        ax.fill_between(t, 0, alt, color=cc, alpha=0.18)
        ax.plot(t, alt, color=cc, lw=1.8)
        ax.set_ylim(bottom=0)
        peak = float(np.nanmax(alt))
        ax.annotate(f"{peak:.0f} m peak",
                    xy=(t[np.nanargmax(alt)], peak),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=cc, fontweight="bold")

        # column title inside top of first panel
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold",
                     color="0.2", pad=8)

        # ── Row 1: Airspeed ───────────────────────────────────────────────────
        ax = axes_grid[1][ci]
        ax.plot(t, arsp, color=cc, lw=1.5, zorder=3)
        if gm.any():
            ax.scatter(t[gm], arsp[gm], s=22, color="#b2182b", zorder=5,
                       edgecolors="white", linewidths=0.5, label="Glide")
        ax.axhline(MIN_AIRSPEED_MS, color="0.6", lw=0.7, ls="--")

        # ── Row 2: Attitude ───────────────────────────────────────────────────
        ax = axes_grid[2][ci]
        ax.axhline(0, color="0.5", lw=0.8, ls=":")
        ax.plot(t, roll,  color="#762a83", lw=1.4, label="Roll")
        ax.plot(t, pitch, color="#1b7837", lw=1.4, label="Pitch", alpha=0.85)
        sym = max(abs(np.nanmax(roll)), abs(np.nanmin(roll)),
                  abs(np.nanmax(pitch)), abs(np.nanmin(pitch)), 5)
        ax.set_ylim(-sym * 1.15, sym * 1.15)

        # ── Row 3: Throttle ───────────────────────────────────────────────────
        ax = axes_grid[3][ci]
        ax.fill_between(t, 0, thr_pct, color="#ca4c07", alpha=0.40)
        ax.plot(t, thr_pct, color="#ca4c07", lw=1.4)
        ax.set_ylim(-5, 105)

        # ── Row 4: Flap ───────────────────────────────────────────────────────
        ax = axes_grid[4][ci]
        if has_flap:
            ax.fill_between(t, 0, flap_pct, color="#2166ac", alpha=0.35)
            ax.plot(t, flap_pct, color="#2166ac", lw=1.4)
            ax.set_ylim(-5, 105)
        else:
            ax.text(0.5, 0.5, "No flap servo\n(C5 not fitted)",
                    ha="center", va="center", fontsize=8,
                    color="0.6", transform=ax.transAxes, style="italic")
            ax.set_ylim(0, 1)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        # x-axis label only on bottom row
        axes_grid[4][ci].set_xlabel("Time from liftoff (s)", fontsize=10)

        # x-ticks only on bottom two rows; others hidden
        for row in range(NROWS - 2):
            axes_grid[row][ci].tick_params(labelbottom=False)

    # ── Shared y-axis labels (left-most column only) ──────────────────────────
    for row, ylabel in enumerate(row_labels):
        axes_grid[row][0].set_ylabel(ylabel, fontsize=10, labelpad=4)

    # ── Shared legend at bottom ───────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor="#f4a582", alpha=0.65, label="Motor on"),
        Patch(facecolor="#b2d8f7", alpha=0.65, label="Flap deployed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b2182b",
               markersize=7, label="Glide segment"),
        Line2D([0], [0], color="#762a83", lw=2, label="Roll"),
        Line2D([0], [0], color="#1b7837", lw=2, label="Pitch"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0), frameon=True,
               edgecolor="0.8")

    fig.text(0.5, 0.965,
             "Flight Telemetry — All Configurations",
             ha="center", va="top", fontsize=13, fontweight="bold", color="0.15")
    fig.text(0.5, 0.942,
             "Altitude  ·  Airspeed  ·  Attitude  ·  Throttle  ·  Flap — airborne window only",
             ha="center", va="top", fontsize=9.5, color="0.45")

    plt.rcParams.update(plt.rcParamsDefault)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).parent

    log_files = {
        "24":  base / "log_24_UnknownDate.bin",
        "112": base / "log_112_UnknownDate.bin",
        "119": base / "log_119_UnknownDate.bin",
        "128": base / "log_128_UnknownDate.bin",
    }

    for name, p in log_files.items():
        if not p.exists():
            print(f"WARNING: {p.name} not found.", file=sys.stderr)

    # ── Parse ─────────────────────────────────────────────────────────────
    frames = {}
    for name, path in log_files.items():
        if not path.exists():
            continue
        print(f"Parsing log_{name} …")
        raw = parse_log(path)
        frames[name] = build_frame(raw)

    # ── Per-log flight window overrides ───────────────────────────────────
    # Log 119: aircraft was on the ground after ~240 s; hard-cap everything
    if "119" in frames:
        fr  = frames["119"]
        cap = fr["liftoff_t"] + 240.0
        fr["landing_t"]  = min(fr["landing_t"], cap)
        fr["duration_s"] = fr["landing_t"] - fr["liftoff_t"]
        in_window = fr["t"] <= fr["landing_t"]
        fr["is_glide"] = fr["is_glide"] & in_window

    # ── Quick sanity check ────────────────────────────────────────────────
    for name, fr in frames.items():
        n_gl = int(fr["is_glide"].sum())
        print(f"  log_{name}: max alt={fr['max_alt_m']:.1f} m, "
              f"flight={fr['duration_s']:.0f} s "
              f"({fr['duration_s']/60:.1f} min), "
              f"bias={fr['airspeed_bias']:.2f} m/s "
              f"({fr['bias_samples']} pre-liftoff samples), "
              f"glide samples={n_gl}")

    # ── Dataset dictionaries (speeds, sinks) arrays ───────────────────────
    def glide_xy(frame, state_filter=None):
        mask = frame["is_glide"]
        if state_filter is not None:
            mask = mask & (frame["flap_state"] == state_filter)
        ok = mask & np.isfinite(frame["airspeed"]) & np.isfinite(frame["sink_ms"])
        return frame["airspeed"][ok], frame["sink_ms"][ok]

    datasets = {}
    if "24"  in frames:
        datasets["baseline"]   = glide_xy(frames["24"])
    if "112" in frames:
        datasets["fixed_flap"] = glide_xy(frames["112"])
    if "119" in frames:
        datasets["flap_up"]    = glide_xy(frames["119"], state_filter=0)
        datasets["flap_down"]  = glide_xy(frames["119"], state_filter=1)
    if "128" in frames:
        datasets["flap_up_128"] = glide_xy(frames["128"])

    for key, (spd, snk) in datasets.items():
        print(f"  {key}: {spd.size} glide samples")

    # ── Generate figures ───────────────────────────────────────────────────
    print("\nGenerating figures …")

    if "24" in frames:
        plot_avionics_dashboard(frames["24"],
                                log_label="Log 24 – Baseline (no flap)",
                                out_path=base / "fig_dash_log24.png")
    if "112" in frames:
        plot_avionics_dashboard(frames["112"],
                                log_label="Log 112 – Fixed flap (C5 = 1100 µs)",
                                out_path=base / "fig_dash_log112.png")
    if "119" in frames:
        plot_avionics_dashboard(frames["119"],
                                log_label="Log 119 – Pilot-switched flap",
                                out_path=base / "fig_dash_log119.png")
    if "128" in frames:
        plot_avionics_dashboard(frames["128"],
                                log_label="Log 128 – Flap retracted (C5 ≈ 1500 µs)",
                                out_path=base / "fig_dash_log128.png")

    plot_telemetry_overview(frames, out_path=base / "fig_telemetry_overview.png")
    plot_combined_four(frames,      out_path=base / "fig_combined_four.png")

    # ── Pool samples into 3 comparison groups ─────────────────────────────
    def _pool(*keys):
        spds = [datasets[k][0] for k in keys if k in datasets and datasets[k][0].size > 0]
        snks = [datasets[k][1] for k in keys if k in datasets and datasets[k][1].size > 0]
        return (np.concatenate(spds) if spds else np.array([]),
                np.concatenate(snks) if snks else np.array([]))

    group_datasets = {}
    if "baseline" in datasets:
        group_datasets["group_baseline"] = datasets["baseline"]
    group_datasets["group_extended"] = _pool("flap_up", "flap_up_128")
    group_datasets["group_deployed"] = _pool("fixed_flap", "flap_down")

    # ── Group comparison figures (3 groups) ────────────────────────────────
    median_sinks_grp = plot_polar(group_datasets,
                                  out_path=base / "fig5_group_polar.png")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("GLIDE PERFORMANCE SUMMARY  –  3-GROUP BREAKDOWN")
    print("═" * 62)

    base_stall = float(np.nanpercentile(group_datasets["group_baseline"][0], 10)) \
                 if "group_baseline" in group_datasets and group_datasets["group_baseline"][0].size >= 3 \
                 else np.nan
    base_sink  = median_sinks_grp.get("group_baseline", np.nan)

    group_info = [
        ("group_baseline", "A – Baseline (original wing, log 24)"),
        ("group_extended", "B – Extended wing, flap retracted (logs 119 + 128)"),
        ("group_deployed", "C – Flap deployed / fixed (log 112 + log 119 down)"),
    ]

    for key, title in group_info:
        spd, snk = group_datasets.get(key, (np.array([]), np.array([])))
        print(f"\n  {title}")
        print(f"    Glide samples   : {spd.size}")
        if spd.size < 3:
            print("    (too few samples for statistics)")
            continue
        stall = float(np.nanpercentile(spd, 10))
        print(f"    Stall speed     : {stall:.3f} m/s", end="")
        if np.isfinite(base_stall) and key != "group_baseline":
            dstall   = 100 * (stall - base_stall) / base_stall
            goal_met = dstall <= -STALL_REDUCTION_GOAL * 100
            print(f"  ({dstall:+.1f}% vs baseline  "
                  f"{'✓ GOAL MET' if goal_met else f'target ≤{base_stall*(1-STALL_REDUCTION_GOAL):.2f}'})")
        else:
            print()
        med_sink = median_sinks_grp.get(key, np.nan)
        if np.isfinite(med_sink):
            print(f"    Median sink     : {med_sink:.3f} m/s", end="")
            if np.isfinite(base_sink) and key != "group_baseline":
                dsink    = 100 * (med_sink - base_sink) / base_sink
                goal_met = dsink <= -SINK_REDUCTION_GOAL * 100
                print(f"  ({dsink:+.1f}% vs baseline  "
                      f"{'✓ GOAL MET' if goal_met else f'target ≤{base_sink*(1-SINK_REDUCTION_GOAL):.2f}'})")
            else:
                print()

    if np.isfinite(base_stall):
        print(f"\n  Project goals (vs Group A baseline)")
        print(f"  {'─'*55}")
        print(f"  Stall speed −{STALL_REDUCTION_GOAL*100:.0f}%  →  target ≤ {base_stall*(1-STALL_REDUCTION_GOAL):.3f} m/s")
        if np.isfinite(base_sink):
            print(f"  Sink rate   −{SINK_REDUCTION_GOAL*100:.0f}%   →  target median ≤ {base_sink*(1-SINK_REDUCTION_GOAL):.3f} m/s")

    print()


if __name__ == "__main__":
    main()

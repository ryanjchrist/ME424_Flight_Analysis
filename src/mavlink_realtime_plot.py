#!/usr/bin/env python3
"""
Real-time telemetry plots for glide testing with flap monitoring.

Connection priority:
  1. Explicit argument on the command line
  2. Auto-detected USB-C serial port  (/dev/tty.usbmodem*, /dev/cu.usbmodem*)
  3. UDP fallback  udpin:0.0.0.0:14550

MAVLink messages used:
  VFR_HUD              – altitude, airspeed, climb rate
  RC_CHANNELS[_RAW]   – flap switch input   (ch 9)
  SERVO_OUTPUT_RAW    – flap servo output   (servo 9)

Plots (2×2, all relevant to project goals):
  TL  Altitude AGL + airspeed vs time
  TR  Sink rate vs time, shaded by flap state
  BL  Live glide polar  (sink vs airspeed) coloured by flap state
  BR  Airspeed histogram with live p10 stall-proxy marker per flap state
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import deque

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

try:
    from pymavlink import mavutil
except ImportError:
    print("Install pymavlink: pip install pymavlink", file=sys.stderr)
    sys.exit(1)


MAX_POINTS   = 1200
SERIAL_BAUD  = 115200
UDP_DEFAULT  = "udpin:0.0.0.0:14550"

FLAP_COLORS = {
    0:  ("tab:blue",   "Flaps up"),
    1:  ("tab:orange", "Flaps down"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_usbc_port() -> str | None:
    """Return the first USB-serial device found on this machine, or None."""
    patterns = [
        "/dev/tty.usbmodem*",
        "/dev/cu.usbmodem*",
        "/dev/tty.usbserial*",
        "/dev/cu.usbserial*",
    ]
    for pat in patterns:
        found = sorted(glob.glob(pat))
        if found:
            return found[0]
    return None


def is_serial_connection(s: str) -> bool:
    return s.startswith("/dev/") or s.startswith("COM") or ".usb" in s


def pwm_to_state(pwm: float, threshold: float) -> int:
    if not np.isfinite(pwm):
        return -1
    return 1 if pwm >= threshold else 0


def get_flap_pwm(msg) -> float | None:
    """Return channel 9 PWM from RC_CHANNELS, or servo 9 from SERVO_OUTPUT_RAW."""
    mtype = msg.get_type()
    if mtype in ("RC_CHANNELS", "RC_CHANNELS_RAW"):
        return getattr(msg, "chan9_raw", np.nan)
    if mtype == "SERVO_OUTPUT_RAW":
        return getattr(msg, "servo9_raw", np.nan)
    return None


def pwm_to_angle(pwm: float) -> float:
    """SG90: 1000µs=0°, 1500µs=90°, 2000µs=180°. Returns deflection from neutral (°)."""
    return (pwm - 1500.0) / 500.0 * 90.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Live glide-testing plots")
    ap.add_argument(
        "connection", nargs="?", default=None,
        help="pymavlink connection string. Default: auto-detected USB-C, then UDP 14550",
    )
    ap.add_argument("--window-s",       type=float, default=120.0,  help="Rolling time window (s)")
    ap.add_argument("--flap-threshold", type=float, default=1500.0, help="PWM threshold for flap deployed")
    args = ap.parse_args()

    # Resolve connection string
    if args.connection:
        conn = args.connection
    else:
        usbc = find_usbc_port()
        if usbc:
            conn = usbc
            print(f"Auto-detected USB-C device: {conn}")
        else:
            conn = UDP_DEFAULT
            print(f"No USB-C device found – falling back to {conn}")

    print(f"Connecting to {conn!r} ...")
    if is_serial_connection(conn):
        mav = mavutil.mavlink_connection(conn, baud=SERIAL_BAUD)
        print(f"Serial connection at {SERIAL_BAUD} baud")
    else:
        mav = mavutil.mavlink_connection(conn)

    mav.wait_heartbeat()
    print(f"Heartbeat from system {mav.target_system}, component {mav.target_component}")

    # Request data streams from ArduPilot (needed when connecting over USB/serial)
    # EXTRA1 = VFR_HUD, RC_CHANNELS = RC inputs, RAW_CONTROLLER = servo outputs
    for stream_id, rate in [
        (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,        10),  # VFR_HUD
        (mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,   10),  # RC + servo
        (mavutil.mavlink.MAV_DATA_STREAM_RAW_CONTROLLER, 5),  # SERVO_OUTPUT_RAW
    ]:
        mav.mav.request_data_stream_send(
            mav.target_system, mav.target_component,
            stream_id, rate, 1,  # 1 = start streaming
        )
    print("Stream requests sent.")

    t0 = time.time()
    times       = deque(maxlen=MAX_POINTS)
    alt_m       = deque(maxlen=MAX_POINTS)
    airspeed_ms = deque(maxlen=MAX_POINTS)
    sink_ms     = deque(maxlen=MAX_POINTS)
    flap_state  = deque(maxlen=MAX_POINTS)

    latest = dict(alt_m=np.nan, airspeed_ms=np.nan, sink_ms=np.nan, flap_pwm=np.nan)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Live glide telemetry", fontsize=12, fontweight="bold")

    ax_alt    = axes[0, 0]   # altitude + airspeed vs time
    ax_sink   = axes[0, 1]   # sink rate vs time, shaded by flap state
    ax_polar  = axes[1, 0]   # glide polar coloured by flap state
    ax_stall  = axes[1, 1]   # airspeed histogram + p10 stall proxy

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    # TL – altitude & airspeed
    (line_alt,)      = ax_alt.plot([], [], color="tab:blue",  lw=1.8, label="Altitude AGL (m)")
    (line_airspeed,) = ax_alt.plot([], [], color="tab:green", lw=1.8, label="Airspeed (m/s)")
    ax_alt.set_title("Altitude & Airspeed")
    ax_alt.set_xlabel("Time in window (s)")
    ax_alt.set_ylabel("m / m/s")
    ax_alt.legend(loc="upper right", fontsize=8)

    # TR – sink rate vs time
    (line_sink,) = ax_sink.plot([], [], color="tab:orange", lw=1.6, label="Sink rate (m/s)")
    ax_sink.axhline(0.0, color="0.4", ls=":", lw=1)
    ax_sink.set_title("Sink Rate  (shaded by flap state)")
    ax_sink.set_xlabel("Time in window (s)")
    ax_sink.set_ylabel("Sink rate (m/s)")
    _legend_patches = [mpatches.Patch(color=c, label=lbl, alpha=0.35)
                       for c, lbl in FLAP_COLORS.values()]
    ax_sink.legend(handles=_legend_patches, fontsize=7, loc="upper right", ncol=2)

    # BL – live glide polar
    ax_polar.set_title("Glide Polar")
    ax_polar.set_xlabel("Airspeed (m/s)")
    ax_polar.set_ylabel("Sink rate (m/s)")

    # BR – stall proxy histogram
    ax_stall.set_title("Airspeed distribution")
    ax_stall.set_xlabel("Airspeed (m/s)")
    ax_stall.set_ylabel("Count")

    status = fig.text(0.5, 0.01, "Waiting for telemetry...", ha="center", fontsize=10)

    # ── animation update ──────────────────────────────────────────────────────
    def update(_frame):
        new_vfr = False
        try:
            while True:
                msg = mav.recv_match(blocking=True, timeout=0.02)
                if msg is None:
                    break
                mtype = msg.get_type()
                if mtype == "VFR_HUD":
                    latest["alt_m"]       = getattr(msg, "alt",      np.nan)
                    latest["airspeed_ms"] = getattr(msg, "airspeed",  np.nan)
                    latest["sink_ms"]     = -getattr(msg, "climb",    np.nan)
                    new_vfr = True
                else:
                    pwm = get_flap_pwm(msg)
                    if pwm is not None and np.isfinite(pwm):
                        latest["flap_pwm"] = pwm
        except Exception as exc:
            status.set_text(f"USB disconnected – reconnect and restart  ({exc})")
            return

        if new_vfr:
            now = time.time() - t0
            times.append(now)
            alt_m.append(latest["alt_m"])
            airspeed_ms.append(latest["airspeed_ms"])
            sink_ms.append(latest["sink_ms"])
            flap_state.append(pwm_to_state(latest["flap_pwm"], args.flap_threshold))

        if len(times) < 2:
            return

        t_arr      = np.asarray(times,       dtype=float)
        in_window  = t_arr >= (t_arr[-1] - args.window_s)
        t_plot     = t_arr[in_window] - t_arr[in_window][0]
        alt_arr    = np.asarray(alt_m,       dtype=float)[in_window]
        air_arr    = np.asarray(airspeed_ms, dtype=float)[in_window]
        sink_arr   = np.asarray(sink_ms,     dtype=float)[in_window]
        state_arr  = np.asarray(flap_state,  dtype=int)[in_window]

        # TL – altitude & airspeed
        line_alt.set_data(t_plot, alt_arr)
        line_airspeed.set_data(t_plot, air_arr)
        ax_alt.set_xlim(max(0.0, t_plot[0]), max(1.0, t_plot[-1]))
        ax_alt.relim(); ax_alt.autoscale_view(scalex=False, scaley=True)

        # TR – sink rate + flap-state shading
        line_sink.set_data(t_plot, sink_arr)
        ax_sink.set_xlim(max(0.0, t_plot[0]), max(1.0, t_plot[-1]))
        ax_sink.relim(); ax_sink.autoscale_view(scalex=False, scaley=True)
        # redraw shaded regions for flap state
        for coll in ax_sink.collections:
            coll.remove()
        if len(t_plot) > 1:
            i = 0
            while i < len(state_arr):
                s = state_arr[i]
                j = i + 1
                while j < len(state_arr) and state_arr[j] == s:
                    j += 1
                color, _ = FLAP_COLORS.get(int(s), ("tab:gray", ""))
                ax_sink.axvspan(t_plot[i], t_plot[min(j, len(t_plot)-1)],
                                alpha=0.18, color=color, linewidth=0)
                i = j

        # BL – glide polar scatter
        ax_polar.cla()
        ax_polar.set_title("Glide Polar")
        ax_polar.set_xlabel("Airspeed (m/s)")
        ax_polar.set_ylabel("Sink rate (m/s)")
        ax_polar.grid(True, alpha=0.3)
        valid = np.isfinite(air_arr) & np.isfinite(sink_arr)
        if np.any(valid):
            point_colors = [FLAP_COLORS.get(int(st), ("tab:gray", ""))[0]
                            for st in state_arr[valid]]
            ax_polar.scatter(air_arr[valid], sink_arr[valid],
                             c=point_colors, s=22, alpha=0.5, edgecolors="none")
            ax_polar.scatter([air_arr[valid][-1]], [sink_arr[valid][-1]],
                             s=140, facecolors="none", edgecolors="black", lw=1.8, zorder=5)
        # legend
        patches = [mpatches.Patch(color=c, label=lbl)
                   for c, lbl in FLAP_COLORS.values() if c != "tab:gray"]
        ax_polar.legend(handles=patches, fontsize=7, loc="upper right")

        # BR – airspeed histogram + p10 stall proxy per flap state
        ax_stall.cla()
        ax_stall.set_title("Airspeed distribution")
        ax_stall.set_xlabel("Airspeed (m/s)")
        ax_stall.set_ylabel("Count")
        ax_stall.grid(True, alpha=0.3)
        shown_states = sorted(set(state_arr.tolist()))
        for s in shown_states:
            if s == -1:
                continue
            mask = (state_arr == s) & np.isfinite(air_arr)
            if np.sum(mask) < 5:
                continue
            color, label = FLAP_COLORS.get(int(s), ("tab:gray", ""))
            spd = air_arr[mask]
            ax_stall.hist(spd, bins=20, alpha=0.45, color=color, label=label)
            p10 = float(np.nanpercentile(spd, 10))
            ax_stall.axvline(p10, color=color, lw=2.0, ls="--")
            ax_stall.text(p10, ax_stall.get_ylim()[1] * 0.92,
                          f"p10={p10:.1f}", color=color, fontsize=8,
                          ha="right", va="top", rotation=90)
        if shown_states:
            ax_stall.legend(fontsize=8, loc="upper right")

        # status bar
        cur_state = int(state_arr[-1]) if state_arr.size else -1
        _, flap_label = FLAP_COLORS.get(cur_state, (""))
        angle_str = f"  ({pwm_to_angle(latest['flap_pwm']):.0f}°)" if np.isfinite(latest["flap_pwm"]) else ""
        status.set_text(
            f"Airspeed {air_arr[-1]:.1f} m/s   Sink {sink_arr[-1]:.2f} m/s   "
            f"Flap: {flap_label}{angle_str}   ch9={latest['flap_pwm']:.0f}µs   |   {conn}"
        )

    ani = animation.FuncAnimation(fig, update, interval=150, blit=False, cache_frame_data=False)  # noqa: F841 – must stay in scope
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.show()


if __name__ == "__main__":
    main()

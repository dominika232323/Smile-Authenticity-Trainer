from typing import Any

import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame, omega: float) -> pd.DataFrame:
    phases = ["onset", "apex", "offset"]
    all_features_dict = {}

    for phase in phases:
        if phase not in df["smile_phase"].unique():
            phase_features_dict = {f"{phase}_{k}": 0.0 for k in extract_features_for_phase(df.iloc[:1], omega).keys()}
        else:
            df_phase = df[df["smile_phase"] == phase].reset_index(drop=True)
            phase_features_raw = extract_features_for_phase(df_phase, omega)
            phase_features_dict = {f"{phase}_{k}": v for k, v in phase_features_raw.items()}

        all_features_dict.update(phase_features_dict)

    return pd.DataFrame([all_features_dict])


def extract_features_for_phase(df_phase: pd.DataFrame, omega: float) -> dict[str, float | int]:
    D = df_phase["D"].to_numpy(dtype=float)
    V = df_phase["V"].to_numpy(dtype=float)
    A = df_phase["A"].to_numpy(dtype=float)

    D_plus_segs, D_minus_segs = segment_increasing_decreasing(D)
    V_plus_segs, V_minus_segs = segment_increasing_decreasing(V)
    A_plus_segs, A_minus_segs = segment_increasing_decreasing(A)

    D_plus = join_segments(D_plus_segs)
    D_minus = join_segments(D_minus_segs)
    V_plus = join_segments(V_plus_segs)
    V_minus = join_segments(V_minus_segs)
    A_plus = join_segments(A_plus_segs)
    A_minus = join_segments(A_minus_segs)

    eta_D = len(D)
    eta_D_plus = len(D_plus)
    eta_D_minus = len(D_minus)

    sum_D_plus = safe_sum(D_plus)
    sum_D_minus = safe_sum(np.abs(D_minus))

    combined_amplitude = sum_D_plus + sum_D_minus

    return {
        "duration_plus": eta_D_plus / omega,
        "duration_minus": eta_D_minus / omega,
        "duration_all": eta_D / omega,
        "duration_ratio_plus": eta_D_plus / eta_D if eta_D else 0,
        "duration_ratio_minus": eta_D_minus / eta_D if eta_D else 0,
        "max_amplitude": safe_max(D),
        "mean_amplitude": safe_mean(D),
        "mean_amplitude_plus": safe_mean(D_plus),
        "mean_amplitude_minus": safe_mean(np.abs(D_minus)),
        "std_amplitude": safe_std(D),
        "total_amplitude_plus": sum_D_plus,
        "total_amplitude_minus": sum_D_minus,
        "net_amplitude": sum_D_plus - sum_D_minus,
        "amplitude_ratio_plus": sum_D_plus / combined_amplitude if combined_amplitude != 0 else 0,
        "amplitude_ratio_minus": sum_D_minus / combined_amplitude if combined_amplitude != 0 else 0,
        "max_speed_plus": safe_max(V_plus),
        "max_speed_minus": safe_max(np.abs(V_minus)),
        "mean_speed_plus": safe_mean(V_plus),
        "mean_speed_minus": safe_mean(np.abs(V_minus)),
        "max_acceleration_plus": safe_max(A_plus),
        "max_acceleration_minus": safe_max(np.abs(A_minus)),
        "mean_acceleration_plus": safe_mean(A_plus),
        "mean_acceleration_minus": safe_mean(np.abs(A_minus)),
        "net_amplitude_duration_ratio": (sum_D_plus - sum_D_minus) * omega / eta_D if eta_D else 0,
    }


def segment_increasing_decreasing(signal: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    D_plus, D_minus = [], []
    current = [signal[0]]
    trend = None

    for i in range(1, len(signal)):
        prev, curr = signal[i - 1], signal[i]

        if curr > prev:
            if trend in (None, "+"):
                trend = "+"
                current.append(curr)
            else:
                D_minus.append(np.array(current))
                current = [prev, curr]
                trend = "+"
        elif curr < prev:
            if trend in (None, "-"):
                trend = "-"
                current.append(curr)
            else:
                D_plus.append(np.array(current))
                current = [prev, curr]
                trend = "-"
        else:
            if trend == "+":
                D_plus.append(np.array(current))
            elif trend == "-":
                D_minus.append(np.array(current))

            current = [curr]
            trend = None

    if trend == "+":
        D_plus.append(np.array(current))
    elif trend == "-":
        D_minus.append(np.array(current))

    return D_plus, D_minus


def join_segments(segments_list: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(segments_list) if len(segments_list) else np.array([])


def safe_len(array: np.ndarray[Any, Any]) -> int:
    return len(array)


def safe_sum(array: np.ndarray[Any, Any]) -> float:
    return float(np.sum(array)) if len(array) else 0.0


def safe_mean(array: np.ndarray[Any, Any]) -> float:
    return float(np.mean(array)) if len(array) else 0.0


def safe_max(array: np.ndarray[Any, Any]) -> float:
    return float(np.max(array)) if len(array) else 0.0


def safe_std(array: np.ndarray[Any, Any]) -> float:
    return float(np.std(array)) if len(array) else 0.0

from __future__ import annotations

import copy
import random
import string
from typing import Any, Sequence

import numpy as np
from numpy import random as np_random

N_RAYS = 9
N_INPUTS = N_RAYS + 1
N_HIDDEN = 12
N_ACTIONS = 4

RAY_NORM = 5.0
SPEED_NORM = 500.0

HEURISTIC_KEYS = [
    "throttle_front_gain", "throttle_flank_gain", "throttle_speed_penalty", "throttle_bias",
    "brake_speed_gain", "brake_front_relief", "brake_bias",
    "steer_gain", "steer_bias",
    "corner_front_thresh", "commit_frames", "safe_speed_frac_per_clearance",
]
HEURISTIC_DEFAULTS = {
    "throttle_front_gain": 5.0, "throttle_flank_gain": 1.0,
    "throttle_speed_penalty": 1.2, "throttle_bias": 1.1,
    "brake_speed_gain": 3.0, "brake_front_relief": 10.0, "brake_bias": -1.6,
    "steer_gain": 4.0, "steer_bias": -0.25,
    "corner_front_thresh": 0.42, "commit_frames": 4.0,
    "safe_speed_frac_per_clearance": 1.35,
}
MUTATION_COUNT_KEYS = ["heuristic", "W1", "b1", "W2", "b2", "trust"]


class AIbrain_Vedant:

    def __init__(self) -> None:
        super().__init__()
        self.score = 0.0
        self.chars = string.ascii_letters + string.digits

        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0

        self.frame_count = 0

        self.init_param()

    def init_param(self) -> None:
        self.heuristic = dict(HEURISTIC_DEFAULTS)
        self._steer_ray_weights = np.array(
            [-0.6, -1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0, 0.6]
        )

        self.W1 = np_random.randn(N_HIDDEN, N_INPUTS) * 0.5
        self.b1 = np.zeros(N_HIDDEN)
        self.W2 = np_random.randn(N_ACTIONS, N_HIDDEN) * 0.5
        self.b2 = np.zeros(N_ACTIONS)
        self.trust = np.full(N_ACTIONS, 0.05)

        self._mutation_counts = {key: 0 for key in MUTATION_COUNT_KEYS}

        self.NAME = "Vedant_" + "".join(random.choices(self.chars, k=5))

        self._reset_episode_state()
        self.store()

    def _reset_episode_state(self) -> None:
        self.commit_direction = 0
        self.commit_frames_left = 0
        self._pos_buf = np.zeros((24, 2), dtype=float)
        self._pos_head = 0
        self._pos_count = 0
        self.stuck_frames = 0
        self.escape_direction = 0
        self.escape_hold = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def g_h(self, key: str) -> float:
        return float(self.heuristic[key])

    def _record_position(self) -> None:
        idx = self._pos_head % self._pos_buf.shape[0]
        self._pos_buf[idx, 0] = self.x
        self._pos_buf[idx, 1] = self.y
        self._pos_head += 1
        self._pos_count = min(self._pos_count + 1, self._pos_buf.shape[0])

    def _recent_displacement(self, window: int):
        window = int(max(1, min(window, self._pos_buf.shape[0], self._pos_count - 1)))
        if self._pos_count <= window:
            return None
        newest = self._pos_buf[(self._pos_head - 1) % self._pos_buf.shape[0]]
        oldest = self._pos_buf[(self._pos_head - 1 - window) % self._pos_buf.shape[0]]
        return float(np.hypot(newest[0] - oldest[0], newest[1] - oldest[1]))

    def _extract_features(self, rays: np.ndarray, speed_frac: float) -> dict[str, float]:
        front = float(rays[4])
        right_space = float(np.dot(rays[:4], -self._steer_ray_weights[:4]))
        left_space = float(np.dot(rays[5:], self._steer_ray_weights[5:]))
        return {
            "front": front,
            "left_space": left_space,
            "right_space": right_space,
            "speed_frac": speed_frac,
        }

    def _heuristic_logits(self, rays: np.ndarray, feats: dict[str, float]) -> np.ndarray:
        h = self.heuristic
        flank_avg = 0.5 * (feats["left_space"] + feats["right_space"])

        throttle = (
            h["throttle_front_gain"] * feats["front"]
            + h["throttle_flank_gain"] * flank_avg
            - h["throttle_speed_penalty"] * feats["speed_frac"]
            + h["throttle_bias"]
        )
        brake = (
            h["brake_speed_gain"] * feats["speed_frac"]
            - h["brake_front_relief"] * feats["front"]
            + h["brake_bias"]
        )
        steer_signal = h["steer_gain"] * float(np.dot(rays, self._steer_ray_weights))
        left = steer_signal + h["steer_bias"]
        right = -steer_signal + h["steer_bias"]

        return np.array([throttle, brake, left, right], dtype=float)

    def decide(self, data: Sequence[float]) -> np.ndarray:
        self.frame_count += 1

        rays = np.asarray(data, dtype=float).ravel()
        if rays.size < N_RAYS:
            rays = np.concatenate([rays, np.zeros(N_RAYS - rays.size)])
        elif rays.size > N_RAYS:
            rays = rays[:N_RAYS]
        rays_norm = np.clip(rays / RAY_NORM, 0.0, 1.0)
        speed_frac = float(np.clip(self.speed / SPEED_NORM, 0.0, 1.0))

        x = np.empty(N_INPUTS)
        x[:N_RAYS] = rays_norm
        x[N_RAYS] = speed_frac

        feats = self._extract_features(rays_norm, speed_frac)
        base_logits = self._heuristic_logits(rays_norm, feats)

        hidden = np.tanh(self.W1.dot(x) + self.b1)
        residual = self.W2.dot(hidden) + self.b2

        logits = base_logits + self.trust * residual
        out = self._sigmoid(logits)

        proposed_direction = 1 if out[2] >= out[3] else -1
        in_corner = feats["front"] < self.g_h("corner_front_thresh")

        if self.escape_hold > 0:
            steer_direction = self.escape_direction
            self.escape_hold -= 1
        elif self.commit_frames_left > 0:
            steer_direction = self.commit_direction
            self.commit_frames_left -= 1
        elif in_corner:
            steer_direction = proposed_direction
            self.commit_direction = proposed_direction
            self.commit_frames_left = int(self.g_h("commit_frames"))
        else:
            steer_direction = proposed_direction
            self.commit_direction = 0

        out = out.copy()
        if steer_direction == 1:
            out[2], out[3] = max(out[2], 0.9), 0.0
        else:
            out[2], out[3] = 0.0, max(out[3], 0.9)

        displacement = self._recent_displacement(10)
        if displacement is not None and displacement < 3.0 and feats["front"] < self.g_h("corner_front_thresh"):
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        if self.stuck_frames > 12 and self.escape_hold <= 0:
            self.escape_direction = -steer_direction
            self.escape_hold = 15
            self.commit_direction = 0
            self.commit_frames_left = 0
            self.stuck_frames = 0

        max_safe_speed_frac = self.g_h("safe_speed_frac_per_clearance") * feats["front"]
        if speed_frac > max_safe_speed_frac:
            out[0] = 0.0
            if speed_frac > max_safe_speed_frac * 1.25:
                out[1] = 1.0

        if self.speed <= 1e-6:
            out = out.copy()
            out[0] = 1.0
            out[1] = 0.0

        return out

    def _decayed_sigma(self, name: str, base: float, floor: float, rate: float = 0.02) -> float:
        self._mutation_counts[name] += 1
        count = self._mutation_counts[name]
        return max(floor, base / (1.0 + rate * count))

    def mutate(self) -> None:
        mutate_rate = 0.4

        h_sigma = self._decayed_sigma("heuristic", base=0.25, floor=0.02)
        for key in self.heuristic:
            if np_random.rand() < mutate_rate:
                self.heuristic[key] += np_random.randn() * h_sigma * max(abs(self.heuristic[key]), 0.5)
        self.heuristic["corner_front_thresh"] = float(np.clip(self.heuristic["corner_front_thresh"], 0.15, 0.85))
        self.heuristic["commit_frames"] = float(np.clip(self.heuristic["commit_frames"], 2.0, 14.0))
        self.heuristic["safe_speed_frac_per_clearance"] = float(
            np.clip(self.heuristic["safe_speed_frac_per_clearance"], 0.5, 3.0)
        )

        for name, base_sigma, floor in (
            ("W1", 0.25, 0.04), ("b1", 0.25, 0.04),
            ("W2", 0.25, 0.04), ("b2", 0.25, 0.04),
        ):
            arr = getattr(self, name)
            sigma = self._decayed_sigma(name, base_sigma, floor)
            mask = np_random.rand(*arr.shape) < mutate_rate
            noise = np_random.randn(*arr.shape) * sigma
            setattr(self, name, arr + noise * mask)

        trust_sigma = self._decayed_sigma("trust", base=0.06, floor=0.01)
        trust_mask = np_random.rand(*self.trust.shape) < mutate_rate
        self.trust = np.clip(
            self.trust + np_random.randn(*self.trust.shape) * trust_sigma * trust_mask,
            0.0, 3.0,
        )

        if np_random.rand() < 0.05:
            target = random.choice(["W1", "W2"])
            arr = getattr(self, target)
            setattr(self, target, arr + np_random.randn(*arr.shape) * 0.4)

        self.NAME += "_m" + "".join(random.choices(self.chars, k=3))

        self._reset_episode_state()
        self.store()

    def store(self) -> None:
        heuristic_values = np.array([self.heuristic[k] for k in HEURISTIC_KEYS], dtype=float)
        mutation_count_values = np.array(
            [self._mutation_counts[k] for k in MUTATION_COUNT_KEYS], dtype=np.int64
        )

        self.parameters = {
            "heuristic_values": heuristic_values,
            "W1": np.array(self.W1, dtype=float, copy=True),
            "b1": np.array(self.b1, dtype=float, copy=True),
            "W2": np.array(self.W2, dtype=float, copy=True),
            "b2": np.array(self.b2, dtype=float, copy=True),
            "trust": np.array(self.trust, dtype=float, copy=True),
            "mutation_count_values": mutation_count_values,
            "NAME": np.array(self.NAME),
        }

    def get_parameters(self) -> dict[str, Any]:
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters: Any) -> None:
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict

        if "heuristic_values" in params_dict:
            values = np.asarray(params_dict["heuristic_values"], dtype=float).ravel()
            self.heuristic = dict(HEURISTIC_DEFAULTS)
            for key, val in zip(HEURISTIC_KEYS, values):
                self.heuristic[key] = float(val)
        elif "heuristic" in params_dict:
            loaded = params_dict["heuristic"]
            loaded = dict(loaded.item() if hasattr(loaded, "item") else loaded)
            self.heuristic = {k: float(loaded.get(k, v)) for k, v in HEURISTIC_DEFAULTS.items()}
        else:
            self.heuristic = dict(HEURISTIC_DEFAULTS)

        self.W1 = np.array(self.parameters["W1"], dtype=float)
        self.b1 = np.array(self.parameters["b1"], dtype=float)
        self.W2 = np.array(self.parameters["W2"], dtype=float)
        self.b2 = np.array(self.parameters["b2"], dtype=float)
        self.trust = np.array(self.parameters.get("trust", np.full(N_ACTIONS, 0.05)), dtype=float)

        if "mutation_count_values" in params_dict:
            values = np.asarray(params_dict["mutation_count_values"]).ravel()
            self._mutation_counts = {key: 0 for key in MUTATION_COUNT_KEYS}
            for key, val in zip(MUTATION_COUNT_KEYS, values):
                self._mutation_counts[key] = int(val)
        elif "mutation_counts" in params_dict:
            loaded_counts = params_dict["mutation_counts"]
            self._mutation_counts = dict(
                loaded_counts.item() if hasattr(loaded_counts, "item") else loaded_counts
            )
        else:
            self._mutation_counts = {key: 0 for key in MUTATION_COUNT_KEYS}

        self.NAME = str(np.asarray(self.parameters["NAME"]).item()) if hasattr(
            np.asarray(self.parameters["NAME"]), "item"
        ) else str(self.parameters["NAME"])

        self._reset_episode_state()
        self.store()

    def calculate_score(self, distance: float, time: float, no: int) -> None:
        avg_speed_bonus = distance / max(time, 0.05)
        self.score = distance + 0.1 * avg_speed_bonus

    def passcardata(self, x: float, y: float, speed: float) -> None:
        self.x = x
        self.y = y
        self.speed = speed
        self._record_position()

    def getscore(self) -> float:
        return self.score


if __name__ == "__main__":
    brain = AIbrain_Vedant()
    print(brain.NAME)
    print(brain.decide(np.array([5.0] * 9).tolist()))

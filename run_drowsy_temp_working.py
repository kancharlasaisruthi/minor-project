"""
run_drowsy.py  — place in example_train\
------------------------------------------
Uses HumanInTheLoopEnv (original, unchanged).
Drowsiness detection and lane-change logic is handled HERE,
wrapping the action before passing it to env.step().

No changes needed to any env file.

Usage:
    python run_drowsy.py
"""

import os
# Set OpenMP environment variable before any library that may load OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import OrderedDict
from enum import Enum, auto

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "env_gym"),
    os.path.join(PROJECT_ROOT, "utils"),
    os.path.join(PROJECT_ROOT, "networks"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Use original HumanInTheLoopEnv — no changes to env files needed
from gym_metadrivepvp_data import HumanInTheLoopEnv
from mlp import StochaPolicy
from metadrive.component.static_object.traffic_object import TrafficObject

# ── Optional sound ────────────────────────────────────────────────────────────
try:
    import pygame
    _PYGAME = True
except ImportError:
    _PYGAME = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PRETRAINED_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "results",
    "DSAC_V2_PVP_RL_gym_metadrivepvp",
    "260128-222729", "apprfunc", "apprfunc_67000.pkl"
)

LANDMARK_PATH          = os.path.join(PROJECT_ROOT, "env_gym",
                                       "shape_predictor_68_face_landmarks.dat")
NUM_EPISODES           = 5
MAX_STEPS              = 2000
DROWSY_CONFIRM_SECONDS = 2.0    # seconds before override activates
STEER_P_GAIN           = 0.35   # proportional gain for lane change
STEER_MAX              = 0.25  # max steering magnitude
STEER_RAMP_SECONDS     = 2.0    # ramp steering from 0→full over this time
LANE_CHANGE_SPEED      = 18.0   # km/h — slow down to this during lane change
BRAKE_ACCEL            = -0.3   # deceleration after reaching lane (gentler braking)
STOP_SPEED             = 2.0    # km/h — considered stopped below this
BEEP_INTERVAL          = 2.0    # seconds between alert beeps

# ─────────────────────────────────────────────────────────────────────────────
#  Drowsiness state machine
# ─────────────────────────────────────────────────────────────────────────────

class DrowsyState(Enum):
    IDLE          = auto()
    CONFIRMING    = auto()
    LANE_CHANGING = auto()
    STRAIGHTENING = auto()
    BRAKING       = auto()
    STOPPED       = auto()


class DrowsinessController:
    """
    Standalone drowsiness controller.
    Call .get_action(rl_action, env) every step.
    Returns the action to actually send to env.step().
    """

    def __init__(self, policy=None):
        self.state               = DrowsyState.IDLE
        self._confirm_start      = 0.0
        self._lane_change_start  = 0.0
        self._last_beep          = 0.0
        self._beep_sound         = None
        self._detector           = None
        self._policy             = policy
        self._target_lane        = None
        self._target_lane_id     = None
        self._stopping_point_added = False
        self.lane_change_times   = []
        self.drowsy_events       = 0
        self._setup_sound()
        self._setup_detector()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_detector(self):
        try:
            from drowsiness_detector import DrowsinessDetector
            self._detector = DrowsinessDetector(
                ear_threshold = 0.25,
                consec_frames = 20,
                camera_index  = 0,
                landmark_path = LANDMARK_PATH,
                show_window   = True,
            )
            self._detector.start()
            print("[Drowsy] Detector started.")
        except Exception as e:
            print(f"[Drowsy] Could not start detector: {e}")
            self._detector = None

    def _setup_sound(self):
        if not _PYGAME:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            n    = int(44100 * 0.7)
            t    = np.linspace(0, 0.7, n, endpoint=False)
            half = n // 2
            wave = np.zeros(n, dtype=np.int16)
            wave[:half] = (np.sin(2 * np.pi * 880       * t[:half]) * 28000).astype(np.int16)
            wave[half:] = (np.sin(2 * np.pi * 880 * 1.3 * t[half:]) * 28000).astype(np.int16)
            self._beep_sound = pygame.sndarray.make_sound(np.column_stack([wave, wave]))
        except Exception as e:
            print(f"[Drowsy] Sound init failed: {e}")

    def reset(self):
        self.state              = DrowsyState.IDLE
        self._confirm_start     = 0.0
        self._lane_change_start = 0.0
        self._target_lane       = None
        self._target_lane_id    = None
        self._stopping_point_added = False
        # Reset per-episode metrics
        self.lane_change_times   = []
        self.drowsy_events       = 0
        if self._detector:
            self._detector.reset()
        self._stop_sound()

    def stop(self):
        if self._detector:
            self._detector.stop()

    # ── Main method called every step ─────────────────────────────────────────

    def get_action(self, rl_action, env):
        """
        Returns (action_to_apply, override_active, state_name, ear_value).
        env is the HumanInTheLoopEnv instance — used to read vehicle state.
        """
        is_drowsy = self._detector.is_drowsy if self._detector else False
        now       = time.time()

        # Wake-up check — cancel override at any non-IDLE state
        if self.state != DrowsyState.IDLE and not is_drowsy:
            self._reset_to_idle()
            ear = self._detector.current_ear if self._detector else 1.0
            return rl_action, False, self.state.name, ear

        # ── IDLE ──────────────────────────────────────────────────────────────
        if self.state == DrowsyState.IDLE:
            if is_drowsy:
                self.state          = DrowsyState.CONFIRMING
                self._confirm_start = now
                print(f"[Drowsy] Signal — confirming for {DROWSY_CONFIRM_SECONDS}s ...")
            ear = self._detector.current_ear if self._detector else 1.0
            return rl_action, False, self.state.name, ear

        # ── CONFIRMING ────────────────────────────────────────────────────────
        if self.state == DrowsyState.CONFIRMING:
            if now - self._confirm_start >= DROWSY_CONFIRM_SECONDS:
                self.state              = DrowsyState.LANE_CHANGING
                self._lane_change_start = now
                self.drowsy_events += 1
                pos = tuple(env.agent.position)
                print(f"[Drowsy] CONFIRMED — emergency lane change! Detection point: {pos}")
                self._play_beep(force=True)
            ear = self._detector.current_ear if self._detector else 1.0
            return rl_action, False, self.state.name, ear   # RL still runs during confirm

        # ── LANE_CHANGING ─────────────────────────────────────────────────────
        if self.state == DrowsyState.LANE_CHANGING:
            self._play_beep()
            if self._target_lane is None:
                self._select_target_lane(env)
            if self._is_in_target_lane(env):
                self.state = DrowsyState.STRAIGHTENING
                print("[Drowsy] Reached target lane — straightening.")
            action = self._lane_change_action(env)
            ear = self._detector.current_ear if self._detector else 1.0
            return action, True, self.state.name, ear

        # ── STRAIGHTENING ─────────────────────────────────────────────────────
        if self.state == DrowsyState.STRAIGHTENING:
            self._play_beep()
            if self._is_aligned_to_target(env):
                self.state = DrowsyState.BRAKING
                print("[Drowsy] Aligned to target lane — braking.")
            action, _ = self._straighten_action(env)
            ear = self._detector.current_ear if self._detector else 1.0
            return action, True, self.state.name, ear

        # ── BRAKING ───────────────────────────────────────────────────────────
        if self.state == DrowsyState.BRAKING:
            self._play_beep()
            speed = getattr(env.agent, "speed_km_h", 0.0)
            if speed <= STOP_SPEED:
                if not self._stopping_point_added:
                    self._stopping_point_added = True
                self.lane_change_times.append(now - self._lane_change_start)
                self.state = DrowsyState.STOPPED
                print("[Drowsy] STOPPED.")
            steer = self._heading_correction(env)
            ear = self._detector.current_ear if self._detector else 1.0
            return [steer, BRAKE_ACCEL], True, self.state.name, ear

        # ── STOPPED ───────────────────────────────────────────────────────────
        if self.state == DrowsyState.STOPPED:
            self._play_beep()
            steer = self._heading_correction(env)
            ear = self._detector.current_ear if self._detector else 1.0
            return [steer, -1.0], True, self.state.name, ear

        ear = self._detector.current_ear if self._detector else 1.0
        return rl_action, False, self.state.name, ear

    # ── Lane change helpers ───────────────────────────────────────────────────

    def _select_target_lane(self, env):
        vehicle = env.agent
        try:
            road_network            = env.engine.current_map.road_network
            road_id, section_id, current_lane_id = vehicle.lane_index
            lanes                  = road_network.graph[road_id][section_id]
            if len(lanes) <= 1:
                self._target_lane = None
                self._target_lane_id = None
                return
            # Always target the rightmost lane for safety
            self._target_lane_id = len(lanes) - 1
            self._target_lane      = lanes[self._target_lane_id]
            print(f"[Drowsy] Target safe lane selected: lane_id={self._target_lane_id} (rightmost)")
        except Exception as e:
            print(f"[Drowsy] Could not select target lane: {e}")
            self._target_lane = None
            self._target_lane_id = None

    def _is_in_target_lane(self, env):
        if self._target_lane is None or self._target_lane_id is None:
            return False
        vehicle = env.agent
        try:
            _, _, lane_id = vehicle.lane_index
            if lane_id != self._target_lane_id:
                return False
            _, lat = self._target_lane.local_coordinates(vehicle.position)
            return abs(lat) < 0.5
        except Exception as e:
            print(f"[Drowsy] _is_in_target_lane failed: {e}")
            return False

    def _is_aligned_to_target(self, env):
        if self._target_lane is None or self._target_lane_id is None:
            return False
        vehicle = env.agent
        try:
            _, _, lane_id = vehicle.lane_index
            if lane_id != self._target_lane_id:
                return False
            long, _ = self._target_lane.local_coordinates(vehicle.position)
            road_heading = self._target_lane.heading_theta_at(long)
            error = road_heading - vehicle.heading_theta
            error = (error + np.pi) % (2 * np.pi) - np.pi
            return abs(np.degrees(error)) < 3.0
        except Exception:
            return False

    def _bias_rl_action(self, rl_action, env, max_bias=0.12):
        if self._target_lane is None or self._target_lane_id is None:
            return rl_action
        action = np.asarray(rl_action, dtype=np.float32)
        steer_bias = self._lane_steering_bias(env, max_bias)
        action[0] = float(np.clip(action[0] + steer_bias, -1.0, 1.0))
        return action.tolist()

    def _lane_steering_bias(self, env, max_bias):
        vehicle = env.agent
        try:
            _, lat = self._target_lane.local_coordinates(vehicle.position)
            return float(np.clip(-0.04 * lat, -max_bias, max_bias))
        except Exception:
            return 0.0

    def _heading_correction(self, env, max_steer=0.2):
        vehicle = env.agent
        try:
            long, _ = self._target_lane.local_coordinates(vehicle.position)
            road_heading = self._target_lane.heading_theta_at(long)
            error = road_heading - vehicle.heading_theta
            error = (error + np.pi) % (2 * np.pi) - np.pi
            return float(np.clip(0.5 * error, -max_steer, max_steer))
        except Exception:
            return 0.0

    def _lane_change_action(self, env):
        vehicle = env.agent
        try:
            elapsed  = time.time() - self._lane_change_start
            ramp     = float(np.clip(elapsed / STEER_RAMP_SECONDS, 0.0, 1.0))
            _, lat = self._target_lane.local_coordinates(vehicle.position)
            steer = float(np.clip(STEER_P_GAIN * lat * ramp, -STEER_MAX, STEER_MAX))
            speed = getattr(vehicle, "speed_km_h", 20.0)
            accel = -0.15 if speed > LANE_CHANGE_SPEED else 0.15
            return [steer, accel]
        except Exception as e:
            print(f"[Drowsy] _lane_change_action failed: {e}")
            speed = getattr(vehicle, "speed_km_h", 20.0)
            accel = -0.15 if speed > LANE_CHANGE_SPEED else 0.15
            return [0.0, accel]

    def _straighten_action(self, env):
        vehicle = env.agent
        try:
            road_network = env.engine.current_map.road_network
            road_id, section_id, lane_id = vehicle.lane_index
            lane = road_network.graph[road_id][section_id][lane_id]
            long, _ = lane.local_coordinates(vehicle.position)
            road_heading = lane.heading_theta_at(long)
            error = road_heading - vehicle.heading_theta
            error = (error + np.pi) % (2 * np.pi) - np.pi
            steer = float(np.clip(0.5 * error, -0.15, 0.15))
            speed = getattr(vehicle, "speed_km_h", 0.0)
            accel = 0.05 if speed < 5.0 else -0.1
            return [steer, accel], abs(np.degrees(error)) < 3.0
        except Exception as e:
            print(f"[Drowsy] _straighten_action failed: {e}")
            return [0.0, -0.1], True

    # ── Sound & reset ─────────────────────────────────────────────────────────

    def _play_beep(self, force=False):
        if not self._beep_sound:
            return
        now = time.time()
        if force or now - self._last_beep >= BEEP_INTERVAL:
            try:
                self._beep_sound.play()
            except Exception:
                pass
            self._last_beep = now

    def _stop_sound(self):
        if self._beep_sound:
            try:
                self._beep_sound.stop()
            except Exception:
                pass

    def _reset_to_idle(self):
        prev = self.state
        self._stop_sound()
        self.state = DrowsyState.IDLE
        if prev != DrowsyState.IDLE:
            print(f"[Drowsy] Driver AWAKE — resuming RL. (was: {prev.name})")


# ─────────────────────────────────────────────────────────────────────────────
#  Load pretrained policy
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(model_path, obs_dim, act_dim, act_high_lim, act_low_lim):
    print(f"[RunDrowsy] Loading: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Extract policy_rl.* keys — this is the autonomous driving policy
    # (policy.*        = behaviour cloning policy, mimics human style)
    # (policy_rl.*     = pure RL policy, drives autonomously — this is what we want)
    # (policy_target.* = slow copy used only during training, not needed)
    prefix    = "policy_rl."
    policy_sd = OrderedDict(
        {k[len(prefix):]: v
         for k, v in state_dict.items()
         if k.startswith(prefix) and not k.startswith("policy_rl_target.")}
    )
    print(f"[RunDrowsy] Using policy_rl — keys: {list(policy_sd.keys())[:4]} ...")

    # Keys contain "policy.0.weight" → std_type is mlp_shared
    std_type = "mlp_shared"

    policy = StochaPolicy(**{
        "obs_dim"                : obs_dim,
        "act_dim"                : act_dim,
        "hidden_sizes"           : [256, 256, 256],
        "hidden_activation"      : "gelu",
        "output_activation"      : "linear",
        "std_type"               : std_type,
        "min_log_std"            : -5,
        "max_log_std"            : 2,
        "act_high_lim"           : act_high_lim,
        "act_low_lim"            : act_low_lim,
        "action_distribution_cls": "TanhGaussDistribution",
    })

    policy.load_state_dict(policy_sd, strict=True)
    policy.to(device)
    policy.eval()
    print("[RunDrowsy] Policy loaded.\n")
    return policy, checkpoint


def get_action(policy, obs):
    with torch.no_grad():
        device = next(policy.parameters()).device
        obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(device)
        output  = policy(obs_t)             # shape [1, act_dim*2]
        act_dim = output.shape[-1] // 2
        action  = output[:, :act_dim]       # mean only (deterministic)
        action  = action.squeeze(0).cpu().numpy()
    return np.clip(action, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Drowsy Driver Detection — Pretrained Model Run")
    print("  Close eyes 2 s → car moves right and stops")
    print("  Open eyes       → car resumes driving immediately")
    print("=" * 60 + "\n")

    # ── Create env ────────────────────────────────────────────────────────────
    env = HumanInTheLoopEnv()
    obs = env.reset()

    obs_dim      = obs.shape[0]
    act_dim      = env.action_space.shape[0]
    act_high_lim = env.action_space.high.astype(np.float32)
    act_low_lim  = env.action_space.low.astype(np.float32)
    print(f"[RunDrowsy] obs_dim={obs_dim}  act_dim={act_dim}\n")

    # ── Load policy ───────────────────────────────────────────────────────────
    policy, checkpoint = load_policy(
        PRETRAINED_MODEL_PATH, obs_dim, act_dim,
        act_high_lim, act_low_lim
    )
    env.activate_rl = checkpoint.get("activate_rl", True)

    # ── Drowsiness controller ─────────────────────────────────────────────────
    drowsy = DrowsinessController(policy)

    # ── Metrics ───────────────────────────────────────────────────────────────
    episode_rewards = []
    episode_lengths = []
    successes = 0
    crashes = 0
    out_of_roads = 0
    total_drowsy_events = 0
    all_lane_change_times = []

    # ── Episode loop ──────────────────────────────────────────────────────────
    try:
        for episode in range(NUM_EPISODES):
            if episode > 0:
                obs = env.reset()
            drowsy.reset()
            ep_reward = 0.0
            print(f"─── Episode {episode + 1} ───")

            for step in range(MAX_STEPS):
                # 1. Get RL action from pretrained model
                rl_action = get_action(policy, obs)

                # 2. Drowsiness controller may override the action
                action, override, state_name, ear = drowsy.get_action(rl_action, env)

                # 3. Step the original env
                obs, reward, done, info = env.step(action)
                ep_reward += reward

                if step % 200 == 0:
                    print(f"  step={step:4d} | "
                          f"drowsy={state_name:14s} | "
                          f"EAR={ear:.3f} | "
                          f"override={override} | "
                          f"reward={reward:+.3f}")
                if done:
                    print(f"  Episode ended step={step}. "
                          f"Reward={ep_reward:.2f}\n")
                    break
            else:
                print(f"  Max steps. Reward={ep_reward:.2f}\n")

            # Collect metrics
            episode_rewards.append(ep_reward)
            episode_lengths.append(step + 1)
            is_success = True
            if done and any(k in str(info) for k in ['crash_vehicle', 'crash_object', 'crash_human']):
                crashes += 1
                is_success = False
            if done and 'out_of_road' in str(info):
                out_of_roads += 1
                is_success = False
            if is_success:
                successes += 1
            total_drowsy_events += drowsy.drowsy_events
            all_lane_change_times.extend(drowsy.lane_change_times)

        # ── Performance Summary ────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  OVERALL PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {NUM_EPISODES}")
        print(f"Success Rate (Completed Episodes): {successes / NUM_EPISODES * 100:.2f}%")
        print(f"Crash Rate: {crashes / NUM_EPISODES * 100:.2f}%")
        print(f"Out of Road Rate: {out_of_roads / NUM_EPISODES * 100:.2f}%")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.2f} steps")
        print(f"Total Drowsy Events: {total_drowsy_events}")
        if all_lane_change_times:
            print(f"Average Lane Change Time: {np.mean(all_lane_change_times):.2f}s ± {np.std(all_lane_change_times):.2f}s")
            print(f"Min/Max Lane Change Time: {np.min(all_lane_change_times):.2f}s / {np.max(all_lane_change_times):.2f}s")
        else:
            print("No lane changes occurred.")

        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('episode_rewards.png')
        print("Reward plot saved as 'episode_rewards.png'")

    finally:
        drowsy.stop()
        env.close()
        print("[RunDrowsy] Done.")


if __name__ == "__main__":
    main()

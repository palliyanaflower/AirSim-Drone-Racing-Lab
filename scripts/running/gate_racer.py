"""
gate_racer.py
─────────────────────────────────────────────────────────────────────────────
A minimal drone-racing controller for AirSim Drone Racing Lab.

Strategy:
  1. Load level, start race (required before enableApiControl)
  2. Extract ground-truth gate poses via simGetObjectPose()
  3. Sort gates into race order (nearest-neighbour from start)
  4. Enable API control, arm, take off
  5. Fly through each gate center, yaw-aligned to the gate normal
  6. Land

Usage:
  python3 gate_racer.py --level_name Soccer_Field_Easy [--velocity 3.0] [--z_offset 0.0]

Init sequence (must match baseline_racer.py exactly):
  simLoadLevel -> confirmConnection -> sleep -> simStartRace
  -> enableApiControl -> arm -> takeoffAsync
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import math
import time
import numpy as np

import airsimdroneracinglab as airsim


# ── helpers ────────────────────────────────────────────────────────────────────

def quaternion_to_yaw(q):
    """Convert an AirSim Quaternionr to a yaw angle (radians)."""
    siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.atan2(siny_cosp, cosy_cosp)


def ned_distance(p1, p2):
    """Euclidean distance between two AirSim Vector3r points."""
    return math.sqrt(
        (p1.x_val - p2.x_val) ** 2
        + (p1.y_val - p2.y_val) ** 2
        + (p1.z_val - p2.z_val) ** 2
    )


def get_gate_pose_with_retry(client, name, max_trials=10):
    """
    simGetObjectPose can return NaN on the first call.
    See: https://github.com/microsoft/AirSim-Drone-Racing-Lab/issues/38
    """
    for _ in range(max_trials):
        pose = client.simGetObjectPose(name)
        if not math.isnan(pose.position.x_val):
            return pose
        time.sleep(0.1)
    return None


def nearest_neighbour_sort(gates, start_pos):
    """
    Greedily sort gates by nearest-neighbour from start_pos.
    Returns a list of (name, pose) tuples in fly-through order.
    """
    remaining = list(gates)
    ordered = []
    current = start_pos

    while remaining:
        closest_idx = min(
            range(len(remaining)),
            key=lambda i: ned_distance(remaining[i][1].position, current),
        )
        chosen = remaining.pop(closest_idx)
        ordered.append(chosen)
        current = chosen[1].position

    return ordered

class GeometricController:
    def __init__(self):
        self.kx = np.array([4.0, 4.0, 6.0])
        self.kv = np.array([3.0, 3.0, 4.0])
        self.kR = np.array([2.0, 2.0, 2.0])
        self.kW = np.array([0.1, 0.1, 0.1])
        self.mass = 1.0
        self.g = 9.81

    def update(self, state, desired):
        # state: {pos, vel, R (3x3), omega (3)}
        # desired: {pos_d, vel_d, acc_d, yaw_d}

        err_pos = state["pos"] - desired["pos"]
        err_vel = state["vel"] - desired["vel"]

        # desired force
        F_des = (
            -self.kx * err_pos
            - self.kv * err_vel
            + self.mass * desired["acc"]
            + np.array([0, 0, self.mass * self.g])
        )

        # thrust (project onto body z-axis)
        thrust = F_des @ (state["R"] @ np.array([0, 0, 1]))

        # desired body z direction
        z_des = F_des / np.linalg.norm(F_des)
        yaw = desired["yaw"]
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0])
        y_des = np.cross(z_des, x_c)
        y_des /= np.linalg.norm(y_des)
        x_des = np.cross(y_des, z_des)

        R_des = np.column_stack((x_des, y_des, z_des))

        # attitude error
        R_err = 0.5 * (R_des.T @ state["R"] - state["R"].T @ R_des)
        e_R = np.array([R_err[2,1], R_err[0,2], R_err[1,0]])

        # body rate error
        e_W = state["omega"] - np.zeros(3)

        # desired moments (not used directly in AirSim)
        M = -self.kR * e_R - self.kW * e_W

        # Convert to roll/pitch/yaw_rate:
        roll_cmd  = e_R[0] * 5.0
        pitch_cmd = e_R[1] * 5.0
        yaw_rate_cmd = e_R[2] * 5.0

        throttle_cmd = np.clip(thrust / (self.mass * self.g), 0.0, 1.0)

        return throttle_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level_name", type=str, default="Soccer_Field_Easy")
    parser.add_argument("--velocity",   type=float, default=3.0)
    parser.add_argument("--z_offset",   type=float, default=0.0,
                        help="NED z offset at each gate (negative = higher)")
    parser.add_argument("--tier",       type=int,   default=1)
    args = parser.parse_args()

    DRONE_NAME = "drone_1"

    # ── 1. Connect & load level ────────────────────────────────────────────────
    print("[+] Connecting to AirSim ...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    print(f"[+] Loading level: {args.level_name}")
    client.simLoadLevel(args.level_name)
    client.confirmConnection()  # failsafe: reconnect after level reload
    time.sleep(2.0)             # let the environment load completely

    # Controller 
    controller = GeometricController()

    # ── 2. Start race  (MUST come before enableApiControl) ────────────────────
    print(f"[+] Starting race (tier={args.tier}) ...")
    client.simStartRace(tier=args.tier)

    # ── 3. Discover & sort gates ───────────────────────────────────────────────
    print("[+] Scanning scene objects for gates ...")
    all_objects = client.simListSceneObjects()
    gate_names = sorted([n for n in all_objects if "Gate" in n])

    if not gate_names:
        print("[!] No gate objects found -- check level name.")
        return

    print(f"[+] Found {len(gate_names)} gate(s): {gate_names}")

    gates = []
    for name in gate_names:
        pose = get_gate_pose_with_retry(client, name)
        if pose is None:
            print(f"    [!] {name}: pose returned NaN after retries -- skipping")
            continue
        gates.append((name, pose))
        yaw_deg = math.degrees(quaternion_to_yaw(pose.orientation))
        print(f"    {name:20s}  x={pose.position.x_val:7.2f}  "
              f"y={pose.position.y_val:7.2f}  z={pose.position.z_val:7.2f}  "
              f"yaw={yaw_deg:6.1f}")

    start_pose = client.simGetVehiclePose(vehicle_name=DRONE_NAME)
    ordered_gates = nearest_neighbour_sort(gates, start_pose.position)

    print("\n[+] Gate fly-through order:")
    for i, (name, _) in enumerate(ordered_gates):
        print(f"    {i+1}. {name}")

    # ── 4. Arm & take off ─────────────────────────────────────────────────────
    print("\n[+] Enabling API control and arming ...")
    client.enableApiControl(vehicle_name=DRONE_NAME)
    client.arm(vehicle_name=DRONE_NAME)

    print("[+] Taking off ...")
    client.takeoffAsync(vehicle_name=DRONE_NAME).join()

    # ── 5. Fly through gates ───────────────────────────────────────────────────
    print("\n[+] Flying through gates ...\n")

    try:
        for i, (name, pose) in enumerate(ordered_gates):

            # TODO: Replace with gate pose estimation
            gx = pose.position.x_val
            gy = pose.position.y_val
            gz = pose.position.z_val + args.z_offset

            gate_yaw = quaternion_to_yaw(pose.orientation)
            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(gate_yaw))

            print(f"  -> Gate {i+1}/{len(ordered_gates)}: {name}  "
                  f"target=({gx:.2f}, {gy:.2f}, {gz:.2f})  "
                  f"yaw={math.degrees(gate_yaw):.1f} deg")

            # TODO: Replace with controller
            client.moveToPositionAsync(
                gx, gy, gz,
                velocity=args.velocity,
                yaw_mode=yaw_mode,
                lookahead=-1,
                adaptive_lookahead=1,
                vehicle_name=DRONE_NAME,
            ).join()

            # # geometric controller
            # throttle_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd = controller.update(state, desired)
            # client.moveByAngleThrottleAsync(
            #     pitch_cmd, roll_cmd, throttle_cmd, yaw_rate_cmd, duration=0.03
            # )

            time.sleep(0.2)

    except Exception as e:
        print(f"\n[!] Error during flight: {e}")

    finally:
        # ── 6. land ───────────────────────────────────────────────────────────
        print("\n[+] Landing ...")
        client.landAsync(vehicle_name=DRONE_NAME).join()
        client.disarm(vehicle_name=DRONE_NAME)
        print("[+] Done.")

if __name__ == "__main__":
    main()
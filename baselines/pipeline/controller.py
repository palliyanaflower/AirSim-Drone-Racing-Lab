"""
Geometric Controller for Quadrotor Trajectory Tracking
=======================================================
Implements Lee et al. (2010) geometric control on SO(3).

Inputs:
  - rpg_time_optimal CSV trajectory (t, px, py, pz, vx, vy, vz, ax, ay, az, ...)
  - Current drone state from AirSim (position, velocity, quaternion)

Outputs:
  - throttle  : [0, 1]  collective thrust normalized
  - roll      : radians, desired roll angle
  - pitch     : radians, desired pitch angle
  - yaw       : radians, desired yaw angle

Coordinate convention: NED (North-East-Down), which is what AirSim uses.
  x = North, y = East, z = Down
  Gravity is +z in NED, i.e. g = [0, 0, +9.81] in body axes pointing down.

Usage:
  controller = GeometricController(mass=0.8, max_thrust=20.0)
  controller.load_trajectory("trajectory.csv")

  # In your control loop:
  state = get_airsim_state()   # See DroneState dataclass below
  cmd = controller.compute(state, current_time)
  client.moveByRollPitchYawThrottleAsync(cmd.roll, cmd.pitch, cmd.yaw, cmd.throttle, dt)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DroneState:
    """
    Current state of the drone, as read from AirSim.
    All quantities in NED world frame unless noted.
    """
    position: np.ndarray        # [x, y, z]  (m)
    velocity: np.ndarray        # [vx, vy, vz]  (m/s)
    rotation: Rotation          # scipy Rotation object representing body orientation
    # AirSim gives you a quaternion; build with:
    #   Rotation.from_quat([qx, qy, qz, qw])


@dataclass
class ControlCommand:
    """Output command sent to the competition API."""
    throttle: float   # [0, 1]
    roll: float       # radians
    pitch: float      # radians
    yaw: float        # radians


@dataclass
class TrajectoryPoint:
    """One sample from the rpg_time_optimal CSV."""
    t: float
    position: np.ndarray     # [x, y, z]
    velocity: np.ndarray     # [vx, vy, vz]
    acceleration: np.ndarray # [ax, ay, az]  (desired, from planner)


# ---------------------------------------------------------------------------
# Trajectory loader
# ---------------------------------------------------------------------------

def load_rpg_trajectory(csv_path: str) -> list[TrajectoryPoint]:
    """
    Load a trajectory CSV produced by rpg_time_optimal.

    rpg_time_optimal outputs columns:
        t, p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z,
        [possibly also: j_x, j_y, j_z, yaw, thrust, ...]

    We only need t, position, velocity, acceleration for geometric control.
    The function is tolerant of extra columns.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # strip any whitespace from headers

    # Accept either 't' or 'time' as the time column
    time_col = 't' if 't' in df.columns else 'time'

    points = []
    for _, row in df.iterrows():
        pt = TrajectoryPoint(
            t=row[time_col],
            position=np.array([row['p_x'], row['p_y'], row['p_z']]),
            velocity=np.array([row['v_x'], row['v_y'], row['v_z']]),
            acceleration=np.array([row['a_x'], row['a_y'], row['a_z']]),
        )
        points.append(pt)
    return points


def interpolate_trajectory(points: list[TrajectoryPoint], t: float) -> TrajectoryPoint:
    """
    Linear interpolation between trajectory samples.

    rpg_time_optimal outputs are densely sampled (~1000 Hz equivalent),
    so linear interpolation is accurate enough for our control rate.
    """
    # Clamp to trajectory bounds
    if t <= points[0].t:
        return points[0]
    if t >= points[-1].t:
        return points[-1]

    # Binary search for the bracketing interval
    lo, hi = 0, len(points) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid].t <= t:
            lo = mid
        else:
            hi = mid

    alpha = (t - points[lo].t) / (points[hi].t - points[lo].t)
    return TrajectoryPoint(
        t=t,
        position=points[lo].position + alpha * (points[hi].position - points[lo].position),
        velocity=points[lo].velocity + alpha * (points[hi].velocity - points[lo].velocity),
        acceleration=points[lo].acceleration + alpha * (points[hi].acceleration - points[lo].acceleration),
    )


# ---------------------------------------------------------------------------
# Geometric controller
# ---------------------------------------------------------------------------

class GeometricController:
    """
    Lee et al. (2010) geometric controller on SO(3).

    The key idea: instead of computing roll/pitch errors in Euler angle space
    (which breaks at large angles), we compute the error directly as a rotation
    matrix on SO(3), then extract the correction.

    Three-step control law:
      1. Position/velocity PD  →  desired thrust vector (direction + magnitude)
      2. Desired thrust direction  →  desired rotation matrix R_des
      3. Rotation error on SO(3)  →  desired roll, pitch, yaw
    """

    def __init__(
        self,
        mass: float = 0.8,          # drone mass (kg)
        max_thrust: float = 20.0,   # maximum collective thrust (N)
        gravity: float = 9.81,      # m/s^2

        # Position PD gains
        Kp: np.ndarray = None,      # proportional gain (position error)
        Kv: np.ndarray = None,      # derivative gain (velocity error)

        # Attitude PD gains
        Kr: np.ndarray = None,      # proportional gain (rotation error)
        Kw: np.ndarray = None,      # derivative gain (angular velocity error)

        # Desired heading (yaw) in radians. Keep constant for racing; 
        # set to 0 to always face +x (North).
        desired_yaw: float = 0.0,
    ):
        self.mass = mass
        self.max_thrust = max_thrust
        self.gravity = gravity
        self.desired_yaw = desired_yaw

        # Default gains — tune these for your specific drone model.
        # Start with small Kp/Kv and increase until tracking is tight
        # without oscillation.
        self.Kp = Kp if Kp is not None else np.array([6.0, 6.0, 10.0])
        self.Kv = Kv if Kv is not None else np.array([4.0, 4.0, 6.0])
        self.Kr = Kr if Kr is not None else np.array([8.0, 8.0, 3.0])
        self.Kw = Kw if Kw is not None else np.array([2.5, 2.5, 1.0])

        self.trajectory: list[TrajectoryPoint] = []

    def load_trajectory(self, csv_path: str):
        self.trajectory = load_rpg_trajectory(csv_path)
        print(f"Loaded trajectory: {len(self.trajectory)} points, "
              f"duration {self.trajectory[-1].t:.2f}s")

    def compute(self, state: DroneState, current_time: float) -> ControlCommand:
        """
        Main control computation. Call this at your control loop rate (e.g. 50-100 Hz).

        Parameters
        ----------
        state : DroneState
            Current drone state from AirSim.
        current_time : float
            Wall time since trajectory start (seconds).

        Returns
        -------
        ControlCommand
            throttle, roll, pitch, yaw ready to send to the competition API.
        """
        # --- Step 1: get reference state at current time ---
        ref = interpolate_trajectory(self.trajectory, current_time)

        # --- Step 2: position and velocity errors ---
        pos_err = state.position - ref.position       # e_p
        vel_err = state.velocity - ref.velocity       # e_v

        # --- Step 3: desired acceleration (PD + feedforward) ---
        # In NED: gravity acts in +z direction.
        gravity_vec = np.array([0.0, 0.0, self.gravity])

        # The desired force vector in world frame:
        #   F_des = m * (-Kp*e_p - Kv*e_v + a_ref + g)
        # The +g term compensates for gravity so hover requires zero error.
        F_des = self.mass * (
            - self.Kp * pos_err
            - self.Kv * vel_err
            + ref.acceleration
            + gravity_vec
        )

        # --- Step 4: desired thrust magnitude ---
        # Get current body z-axis (thrust direction) from rotation matrix.
        R_current = state.rotation.as_matrix()   # 3x3, columns are body axes in world frame
        b3_current = R_current[:, 2]             # current body z (thrust direction)

        # Thrust is the projection of F_des onto current body z axis.
        # This is the standard differential flatness result.
        thrust = np.dot(F_des, b3_current)
        thrust = np.clip(thrust, 0.0, self.max_thrust)

        # Normalized throttle [0, 1]
        throttle = thrust / self.max_thrust

        # --- Step 5: desired rotation matrix R_des ---
        # The desired body z-axis points along F_des (thrust direction).
        b3_des = F_des / (np.linalg.norm(F_des) + 1e-6)

        # The desired body x-axis: we want the drone's nose to point in the
        # direction of the desired_yaw angle projected onto the horizontal plane,
        # then made orthogonal to b3_des.
        heading = np.array([np.cos(self.desired_yaw), np.sin(self.desired_yaw), 0.0])
        b2_des = np.cross(b3_des, heading)
        b2_des = b2_des / (np.linalg.norm(b2_des) + 1e-6)
        b1_des = np.cross(b2_des, b3_des)

        # Construct desired rotation matrix. Columns are body axes expressed in world frame.
        R_des = np.column_stack([b1_des, b2_des, b3_des])

        # --- Step 6: rotation error on SO(3) ---
        # The error rotation is: R_err = R_des^T * R_current
        # The "vee" map extracts the axis-angle vector from the skew-symmetric part.
        R_err = R_des.T @ R_current
        # Rotation error vector (Lee et al. eq. 10): e_R = 0.5 * vee(R_des^T R - R^T R_des)
        e_R = 0.5 * _vee(R_err - R_err.T)

        # Angular velocity error: for now assume desired angular velocity = 0
        # (the planner doesn't give us angular velocity references).
        # This is a simplification; at high speeds you'd want to add a feed-forward
        # term from the trajectory's jerk/snap.
        e_omega = np.zeros(3)

        # --- Step 7: attitude correction torque (conceptual) ---
        # We don't send torques directly; instead we extract the desired attitude angles
        # that the inner loop (Betaflight / competition API) will track.
        # The correction is: tau = -Kr * e_R - Kw * e_omega
        # We use this to nudge R_des slightly, then extract Euler angles.
        # For simplicity (and because the inner loop handles rate control),
        # we just extract roll/pitch/yaw directly from R_des.

        # --- Step 8: extract roll, pitch, yaw from R_des ---
        desired_rotation = Rotation.from_matrix(R_des)
        # Use intrinsic ZYX Euler angles: yaw first, then pitch, then roll.
        # scipy convention: 'ZYX' gives [yaw, pitch, roll] — note the order.
        euler = desired_rotation.as_euler('ZYX')   # [yaw, pitch, roll]
        yaw   = euler[0]
        pitch = euler[1]
        roll  = euler[2]

        return ControlCommand(
            throttle=float(throttle),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        )


# ---------------------------------------------------------------------------
# Helper: vee map
# ---------------------------------------------------------------------------

def _vee(skew: np.ndarray) -> np.ndarray:
    """
    Extract the 3-vector from a 3x3 skew-symmetric matrix.

    The vee map is the inverse of the hat (^) map:
      hat(w) = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
      vee(hat(w)) = w

    Used to convert the rotation error matrix into a rotation error vector.
    """
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


# ---------------------------------------------------------------------------
# AirSim state helper
# ---------------------------------------------------------------------------

def airsim_state_to_drone_state(airsim_client, vehicle_name: str = "") -> DroneState:
    """
    Convert AirSim MultirotorState to our DroneState dataclass.

    AirSim uses NED coordinates, which matches our controller convention.

    Example usage:
        import airsim
        client = airsim.MultirotorClient()
        state = airsim_state_to_drone_state(client)
    """
    import airsim
    ms = airsim_client.getMultirotorState(vehicle_name=vehicle_name)
    kin = ms.kinematics_estimated

    position = np.array([
        kin.position.x_val,
        kin.position.y_val,
        kin.position.z_val,
    ])
    velocity = np.array([
        kin.linear_velocity.x_val,
        kin.linear_velocity.y_val,
        kin.linear_velocity.z_val,
    ])
    q = kin.orientation
    # AirSim quaternion: w, x, y, z  — scipy expects x, y, z, w
    rotation = Rotation.from_quat([q.x_val, q.y_val, q.z_val, q.w_val])

    return DroneState(position=position, velocity=velocity, rotation=rotation)


# ---------------------------------------------------------------------------
# Example main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal example showing how to use the controller with AirSim.
    Replace the AirSim calls with your competition API equivalents.
    """
    import time
    import airsim

    # --- Setup ---
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    controller = GeometricController(
        mass=0.8,           # kg — match your drone model
        max_thrust=20.0,    # N  — match your drone model
        desired_yaw=0.0,    # radians, face North
    )
    controller.load_trajectory("trajectory.csv")

    # --- Control loop ---
    CONTROL_HZ = 50          # Hz
    dt = 1.0 / CONTROL_HZ
    t_start = time.time()
    t_end = controller.trajectory[-1].t

    print("Starting trajectory tracking...")
    while True:
        t_elapsed = time.time() - t_start
        if t_elapsed > t_end:
            print("Trajectory complete.")
            break

        state = airsim_state_to_drone_state(client)
        cmd = controller.compute(state, t_elapsed)

        client.moveByRollPitchYawThrottleAsync(
            cmd.roll, cmd.pitch, cmd.yaw, cmd.throttle, dt
        )
        time.sleep(dt)
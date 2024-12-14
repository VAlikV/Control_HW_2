import numpy as np
import os
from simulator import Simulator
from pathlib import Path
from typing import Dict
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.linalg import logm

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model_1 = pin.buildModelFromMJCF(xml_path)
data_1 = model_1.createData()

times_1 = []
positions_1 = []
velocities_1 = []
error_1 = []
control_1 = []
task_space_pos = []
task_space_err = []

X_1 = []
Y_1 = []

def plot_results(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray, torque: np.ndarray, error: np.ndarray, task_space_pos: np.ndarray, task_space_err: np.ndarray):
    """Plot and save simulation results."""
    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/HW_pos.png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/HW_vel.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(torque.shape[1]):
        plt.plot(times, torque[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Torque [Nm]')
    plt.title('Joint Torque over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/HW_tor.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(error.shape[1]):
        plt.plot(times, error[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Error')
    plt.title('Error over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/HW_err.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(task_space_pos.shape[1]):
        plt.plot(times, task_space_pos[:, i])
    plt.xlabel('Time [s]')
    plt.ylabel('Position')
    plt.title('Task space position over Time')
    plt.legend(['X','Y','Z'])
    plt.grid(True)
    plt.savefig('logs/plots/HW_tspos.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(task_space_err.shape[1]):
        plt.plot(times, task_space_err[:, i])
    plt.xlabel('Time [s]')
    plt.ylabel('Error')
    plt.title('Task space error over Time')
    plt.legend(['X','Y','Z'])
    plt.grid(True)
    plt.savefig('logs/plots/HW_tserr.png')
    plt.close()

def skew_to_vector(m):
    return np.array([m[2,1], m[0,2], m[1,0]])

def R_err(Rd, R):
    error_matrix = Rd @ R.T
    error_log = logm(error_matrix)
    return skew_to_vector(error_log)

def traj(t):
    R = 0.15
    X = np.array([0.15+R*np.sin(2*np.pi*t), 0.40+R*np.cos(2*np.pi*t), 0.5, 0, 0., 0])
    dX = np.array([2*np.pi*R*np.cos(2*np.pi*t), -2*np.pi*R*np.sin(2*np.pi*t), 0, 0, 0, 0])
    ddX = np.array([-((2*np.pi)**2)*R*np.sin(2*np.pi*t), -((2*np.pi)**2)*R*np.cos(2*np.pi*t), 0, 0, 0, 0])

    return X, dX, ddX

def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:
    """Example task space controller."""

    desired_position = desired['pos'] # [x_des, y_des, z_des]
    desired_quaternion = desired['quat'] # [w_des, x_des, y_des, z_des]
    desired_quaternion_pin = pin.Quaternion(*desired_quaternion)
    rpy = pin.rpy.matrixToRpy(desired_quaternion_pin.toRotationMatrix())

    X_d = np.hstack((desired_position, rpy))
    dX_d = np.zeros(6)
    ddX_d = np.zeros(6)

    X_d, dX_d, ddX_d = traj(t)

    pin.forwardKinematics(model_1, data_1, q, dq)

    end_effector_id = model_1.getFrameId("end_effector")
    frame = pin.LOCAL_WORLD_ALIGNED

    pin.updateFramePlacement(model_1, data_1, end_effector_id)

    twist = pin.getFrameVelocity(model_1, data_1, end_effector_id, frame)
    # dtwist = pin.getFrameAcceleration(model_1, data_1, end_effector_id, frame)

    J = pin.getFrameJacobian(model_1, data_1, end_effector_id, frame)
    dJ = pin.getFrameJacobianTimeVariation(model_1, data_1, end_effector_id, frame)

    ee_pose = data_1.oMf[end_effector_id]
    end_effector_position = ee_pose.translation
    end_effector_rotation_matrix = ee_pose.rotation

    a_d = X_d[3:]
    R_d = pin.rpy.rpyToMatrix(*a_d)

    angles_err = R_err(R_d, end_effector_rotation_matrix)

    X = np.hstack((X_d[0:3]-end_effector_position, angles_err))

    pin.computeAllTerms(model_1, data_1, q, dq)

    M = data_1.M
    nle = data_1.nle

    kp = np.eye(6) * np.array([100, 100, 100, 100, 100, 100])
    kd = np.eye(6) * np.array([20, 20, 20, 20, 20, 20])

    print(X)

    a_x = kp @ X + kd @ (dX_d - twist.np) + ddX_d

    a_q = np.linalg.pinv(J)@(a_x - dJ@dq)

    tau = M @ a_q + nle 

    # print(tau)
    times_1.append(t)
    positions_1.append(q)
    velocities_1.append(dq)
    error_1.append(X)
    control_1.append(tau)
    task_space_pos.append(end_effector_position)
    task_space_err.append(X_d[0:3]-end_effector_position)

    X_1.append(end_effector_position[0])
    Y_1.append(end_effector_position[1])
    
    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/HW_2.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    sim.set_controller(task_space_controller)
    sim.run(time_limit=10.0)

if __name__ == "__main__":
    # print(current_dir)
    main() 

    times_1 = np.array(times_1)
    positions_1 = np.array(positions_1)
    velocities_1 = np.array(velocities_1)
    control_1 = np.array(control_1)
    error_1 = np.array(error_1)
    task_space_pos = np.array(task_space_pos)
    task_space_err = np.array(task_space_err)

    print(task_space_pos)
    print(task_space_pos[:,0])

    plot_results(times_1, positions_1, velocities_1, control_1, error_1, task_space_pos, task_space_err)

    X_1 = np.array(X_1)
    Y_1 = np.array(Y_1)

    plt.figure(figsize=(10, 6))
    plt.plot(X_1, Y_1)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('X and Y')
    plt.grid(True)
    plt.savefig('logs/plots/HW_XY.png')
    plt.close()
    
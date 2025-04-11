import pybullet as p
import pybullet_data
import time
import csv
import argparse
import os

def load_trajectory(filename):
    """
    Loads a CSV file where each row is a list of joint angles.
    """
    trajectory = []
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        # Read the first row and try to convert its first element to float.
        first_row = next(csvreader, None)
        if first_row is not None:
            try:
                # If conversion works, the row is data, so process it.
                float(first_row[0])
                trajectory.append([float(val) for val in first_row])
            except ValueError:
                # Otherwise, it's a header; skip it.
                pass
        # Process the remaining rows.
        for row in csvreader:
            if row:  # Skip empty rows
                trajectory.append([float(val) for val in row])
    return trajectory

def play_trajectory(robotId, trajectory, num_joints):
    """
    Plays the provided trajectory on the robot.
    Each trajectory configuration should have 'num_joints' values.
    """
    for config in trajectory:
        if len(config) != num_joints:
            print("Skipping configuration:", config, 
                  "because its length does not match the number of joints.")
            continue

        # Set the target position for each joint.
        for joint_index in range(num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=robotId,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=config[joint_index]
            )

        # Step simulation to visualize the motion.
        for _ in range(240):  # simulate for 240 steps (~1 second at 240Hz)
            p.stepSimulation()
            time.sleep(1./10000.)

def main():
    # Define default file paths.
    default_urdf_path = "/home/sahruday/Desktop/ompl/vamp/resources/panda/panda.urdf"
    default_trajectory_path = "/home/sahruday/Desktop/ompl/vamp/solution/BITstar.csv"
    default_folder_path = "/home/sahruday/Desktop/ompl/vamp/solution"

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Visualize robot trajectory using PyBullet."
    )
    parser.add_argument('--urdf', type=str, default=default_urdf_path,
                        help="Path to the robot URDF file. Default: " + default_urdf_path)
    parser.add_argument('--trajectory', type=str, default=default_trajectory_path,
                        help="Path to a trajectory CSV file. Default: " + default_trajectory_path)
    parser.add_argument('--folder', type=str, default=default_folder_path,
                        help="If provided, run all trajectory CSV files in this folder instead of a single file.")
    args = parser.parse_args()

    # Connect to PyBullet in GUI mode.
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For loading plane.urdf, etc.
    p.setGravity(0, 0, -9.8)

    # Optionally load a ground plane.
    planeId = p.loadURDF("plane.urdf")

    # Load the robot from the provided URDF file.
    robotId = p.loadURDF(args.urdf, useFixedBase=True)
    print("Loaded robot from URDF:", args.urdf)

    # Define an array "problem" with sphere centers.
    problem = [
        [0.55, 0, 0.25],
        [0.35, 0.35, 0.25],
        [0, 0.55, 0.25],
        [-0.55, 0, 0.25],
        [-0.35, -0.35, 0.25],
        [0, -0.55, 0.25],
        [0.35, -0.35, 0.25],
        [0.35, 0.35, 0.8],
        [0, 0.55, 0.8],
        [-0.35, 0.35, 0.8],
        [-0.55, 0, 0.8],
        [-0.35, -0.35, 0.8],
        [0, -0.55, 0.8],
        [0.35, -0.35, 0.8]
    ]
    sphere_radius = 0.1

    # Create visual and collision shapes for spheres.
    sphere_visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1, 0, 0, 1]
    )
    sphere_collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE, radius=sphere_radius
    )

    # Add each sphere to the environment.
    for center in problem:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=sphere_collision_shape_id,
            baseVisualShapeIndex=sphere_visual_shape_id,
            basePosition=center
        )

    # Determine the number of joints for the robot.
    num_joints = 7  # Example for a 7-DOF robot like the Panda.

    # List to hold trajectory file paths.
    trajectory_files = []

    if args.folder is not None:
        # Run all CSV files in the provided folder.
        if os.path.isdir(args.folder):
            for file in os.listdir(args.folder):
                if file.endswith('.csv'):
                    trajectory_files.append(os.path.join(args.folder, file))
            trajectory_files.sort()  # Optional: sort files alphabetically
            if not trajectory_files:
                print("No CSV trajectory files found in the folder:", args.folder)
                return
            print("Found", len(trajectory_files), "trajectory files in folder:", args.folder)
        else:
            print("The provided folder path does not exist:", args.folder)
            return
    else:
        # Use a single trajectory file.
        if os.path.isfile(args.trajectory):
            trajectory_files.append(args.trajectory)
            print("Using trajectory file:", args.trajectory)
        else:
            print("The trajectory file does not exist:", args.trajectory)
            return

    # Run the trajectories sequentially.
    for traj_file in trajectory_files:
        print("\nPlaying trajectory from file:", traj_file)
        trajectory = load_trajectory(traj_file)
        print("Loaded trajectory with", len(trajectory), "configurations.")
        play_trajectory(robotId, trajectory, num_joints)
        # Optional pause between trajectories.
        input("Press Enter to continue to the next trajectory (if any)...")

    # Keep the simulation window open until the user decides to exit.
    input("Press Enter to exit the simulation...")
    p.disconnect()

if __name__ == "__main__":
    main()

import pinocchio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from tqdm import tqdm
import copy

# Load the URDF model
urdf_filename = "./urdf/wow_body_little.urdf"
urdf_filename = "./urdf/wow_body_large.urdf"
urdf_filename = "./urdf/panda.urdf"
urdf_filename = "./urdf/ARM002.urdf"

num_samples = 5000

vis_orientation = False
vis_surface = False

model = pinocchio.buildModelFromUrdf(urdf_filename)
data = model.createData()

# 输出每个帧的 ID 和名称
for frame_id in range(model.nframes):
    frame = model.frames[frame_id]
    print(f"Frame ID: {frame_id}, Name: {frame.name}")

# NOTE: 请选择正确的 id
JOINT_ID = 17



# Define joint limits
qup = model.upperPositionLimit.tolist()
qlow = model.lowerPositionLimit.tolist()
qup = np.array(qup)
qlow = np.array(qlow)
print(qup)
print(qlow)

# Initialize lists to collect positions and orientations
positions = []
x_axes = []
y_axes = []
z_axes = []



# Sample joint angles uniformly
for i in tqdm(range(num_samples)):
    random_samples = np.random.rand(len(qlow))

    random_arrays = [qlow + (qup - qlow) * random_samples]
    zero_padding = [np.zeros(model.nq - len(qup))]

    # 合并随机数组和零填充，最后转换为 NumPy 数组
    q = np.concatenate(random_arrays + zero_padding)

    # Compute forward kinematics
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.updateFramePlacements(model, data)

    # Collect the position and orientation for the specific joint (e.g., JOINT_ID = 18)
    pos = data.oMf[JOINT_ID]
    position = pos.translation
    rotation = pos.rotation

    delta = rotation @ np.array([0, 0, -0.2])
    if position[0] > 0:
        positions.append(copy.deepcopy(position) + delta)

    x_axes.append(rotation @ np.array([1, 0, 0]))
    y_axes.append(rotation @ np.array([0, 1, 0]))
    z_axes.append(rotation @ np.array([0, 0, 1]))

# Convert lists to numpy arrays
positions = np.array(positions)
x_axes = np.array(x_axes)
y_axes = np.array(y_axes)
z_axes = np.array(z_axes)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot joint positions
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')

if vis_surface:
    # If you have more than 3D points, create a surface plot
    if len(positions) > 3:
        # Compute Delaunay triangulation for surface plot
        tri = Delaunay(positions[:, :2])  # Triangulate in 2D plane (x, y)

        # Plot the surface
        ax.plot_trisurf(positions[:, 0], positions[:, 1], positions[:, 2], triangles=tri.simplices, cmap='viridis', edgecolor='none')

if vis_orientation:
    # Plot coordinate axes
    for i in range(len(positions)):
        # Plot X axis (red)
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                x_axes[i, 0], x_axes[i, 1], x_axes[i, 2], color='r', length=0.02)
        # Plot Y axis (green)
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                y_axes[i, 0], y_axes[i, 1], y_axes[i, 2], color='g', length=0.02)
        # Plot Z axis (blue)
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                z_axes[i, 0], z_axes[i, 1], z_axes[i, 2], color='b', length=0.02)

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)

plt.show()

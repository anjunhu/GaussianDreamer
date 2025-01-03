import open3d as o3d
import numpy as np

# Load your point cloud
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud("/home/ubuntu/scene/GaussianDreamer/outputs/gaussiandreamer-sd/a_panda_sofa@20241114-205956/save/shape.ply")

# Get points and colors
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# Create a uniform scale (isotropic radius) for all points
# You can adjust this value based on your needs
uniform_scale = np.ones((len(points), 1)) * 0.01  # 0.01 is an example value

# Create full opacity for all points
opacity = np.ones((len(points), 1))

# Combine all attributes
attributes = np.hstack((points, colors, uniform_scale, opacity))

# Create a new point cloud with these attributes
gaussian_pcd = o3d.geometry.PointCloud()
gaussian_pcd.points = o3d.utility.Vector3dVector(points)
gaussian_pcd.colors = o3d.utility.Vector3dVector(colors)

# Add custom attributes for scale and opacity
gaussian_pcd.normals = o3d.utility.Vector3dVector(np.hstack((uniform_scale, opacity)))

# Save as PLY
o3d.io.write_point_cloud("gaussian_splat.ply", gaussian_pcd, write_ascii=True)

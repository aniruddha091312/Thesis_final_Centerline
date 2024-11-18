import numpy as np

import open3d as o3d


# Replace these with your actual data
point_cloud_data = np.random.rand(1000, 3)  # Example point cloud data
centerline_data = np.random.rand(100, 3)    # Example centerline data


# Initialize the ExternalVisualizer
ev = o3d.visualization.ExternalVisualizer()

# Create Open3D PointCloud objects
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
point_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for point cloud

centerline = o3d.geometry.PointCloud()
centerline.points = o3d.utility.Vector3dVector(centerline_data)
centerline.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for centerline

# Send the geometries to the external visualizer
ev.set([point_cloud, centerline])


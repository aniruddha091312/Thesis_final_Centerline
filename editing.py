# Example code to visualize points near centerline
import os

import numpy as np

from data_utils.MeshCenterline import get_point_cloud, dataset_path, pc_normalize, remove_duplicate_points, \
    load_centerline, classify_point, visualize_results_with_centerline

# Set up example point cloud and centerline
file_name = '2023_RCSE_Centerline/AC_20150820_left.obj'  # Replace with an actual sample filename from your dataset
point_cloud = get_point_cloud(os.path.join(dataset_path, file_name))
point_cloud = pc_normalize(point_cloud)

centerline_file_path = os.path.join(dataset_path, file_name.replace('.obj', '_centerline.dat'))
centerline_points = load_centerline(centerline_file_path)
centerline_points = remove_duplicate_points(centerline_points)
centerline_points = pc_normalize(centerline_points)

# Set the threshold for proximity to centerline points
threshold = 0.05

# Run classification of points near the centerline
near_centerline_labels = classify_point(point_cloud, centerline_points, threshold=threshold)

# Visualize the results
visualize_results_with_centerline(
    point_clouds=[point_cloud],
    preds=[near_centerline_labels],  # Using near_centerline_labels as predictions
    labels=[near_centerline_labels],  # Ground truth labels for visualization purposes
    centerline_points=centerline_points,
    threshold=threshold,
    num_samples=1  # Visualize a single sample
)



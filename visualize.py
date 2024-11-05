import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_point_cloud(file_path, num_points=1024):
    mesh = trimesh.load_mesh(file_path)
    point_cloud = mesh.sample(num_points)  # Sample points from the mesh surface
    print(point_cloud.shape)
    return point_cloud.astype(np.float32)

def visualize_point_cloud(point_cloud):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, color='blue')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Normalized Point Cloud Visualization")
    plt.show()

# def visualize_point_cloud(point_cloud):
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, color='blue')
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title("Point Cloud Visualization")
#     plt.show()
#
#
def load_centerline(file_path):
    return np.loadtxt(file_path, skiprows=1, usecols=(0, 1, 2))
#
#
# def check_point_cloud_uniformity(point_cloud):
#     # Calculate distance matrix
#     dists = distance_matrix(point_cloud, point_cloud)
#     avg_dist = np.mean(dists)
#     return avg_dist
#
#
# def visualize_centerline(centerline_points, color='blue'):
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(centerline_points[:, 0], centerline_points[:, 1], centerline_points[:, 2], s=5, color=color)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title("Centerline Visualization")
#     plt.show()
#
#
# def remove_duplicate_points(centerline,eps,  min_samples=2):
#     """
#     Use DBSCAN clustering to remove duplicate or very close centerline points.
#     :param centerline: np.array of centerline points
#     :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood (cluster)
#     :param min_samples: The number of points required to form a dense region (a cluster)
#     :return: np.array of refined centerline points
#     """
#     clustering = DBSCAN( eps, min_samples=min_samples).fit(centerline)
#     unique_points = []
#     for cluster_id in set(clustering.labels_):
#         if cluster_id != -1:  # Ignore noise points
#             cluster_points = centerline[clustering.labels_ == cluster_id]
#             # Use the mean of the cluster points as the representative point
#             cluster_center = np.mean(cluster_points, axis=0)
#             unique_points.append(cluster_center)
#     return np.array(unique_points)


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


file_path = '2023_RCSE_Centerline/AC_20150820_left.obj'
point_cloud = get_point_cloud(file_path)
normalized_point_cloud = pc_normalize(point_cloud)
visualize_point_cloud(normalized_point_cloud)

centerline_path ='2023_RCSE_Centerline/AC_20150820_left_centerline.dat'
centerline_points = load_centerline(centerline_path)
normalized_cp= pc_normalize(centerline_points)
visualize_point_cloud(normalized_cp)
# original_point_cloud = get_point_cloud(file_path, num_points=1024)
# visualize_point_cloud(original_point_cloud)


centerline_points = load_centerline(centerline_path)
# visualize_centerline(centerline_points, color='blue')
# print(f"Number of points after filtering: {centerline_points.shape[0]}")


# Calculate eps based on average point spacing in the centerline
# avg_dist = check_point_cloud_uniformity(centerline_points)
# eps = avg_dist * 0.005# Adjust factor based on clustering needs

# filtered_centerline_points = remove_duplicate_points(centerline_points, eps=eps ,  min_samples=2)
# print(f"Number of points after filtering: {filtered_centerline_points.shape[0]}")
# visualize_centerline(filtered_centerline_points)

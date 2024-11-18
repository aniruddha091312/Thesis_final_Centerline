import os
import warnings
from datetime import time

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree, distance_matrix
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

dataset_path = '2023_RCSE_Centerline'


def get_point_cloud(file_path, num_points=1024):
    mesh = trimesh.load_mesh(file_path)
    point_cloud = mesh.sample(num_points)  # Sample points from the mesh surface
    print(point_cloud.shape)
    return point_cloud.astype(np.float32)


def load_centerline(file_path):
    return np.loadtxt(file_path, skiprows=1, usecols=(0, 1, 2))


def random_scale(pc, scale_low=0.8, scale_high=1.2):
    scale = np.random.uniform(scale_low, scale_high)
    return pc * scale


def random_jitter(pc, sigma=0.01, clip=0.05):
    jitter = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
    return pc + jitter


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def check_point_cloud_uniformity(point_cloud):
    # Calculate distance matrix
    dists = distance_matrix(point_cloud, point_cloud)
    avg_dist = np.mean(dists)
    return avg_dist


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def classify_point(point_cloud, centerline_points, threshold=0.05):
    """
#     Classify points in the point cloud based on proximity to centerline points.
#
#     Args:
#     - point_cloud: Numpy array of points (Nx3) representing the point cloud.
#     - centerline_points: Numpy array of points (Mx3) representing the centerline.
#     - threshold: The maximum distance to consider a point "close" to the centerline.
#
#     Returns:
#     - classifications: Numpy array of size N, where each element is 0 or 1 depending on proximity to centerline.
#     """
    tree = cKDTree(point_cloud)
    classifications = np.zeros(len(point_cloud))
    for point in centerline_points:
        distance, idx = tree.query(point)

        # Check if the nearest point is within the specified threshold distance
        if distance <= threshold:
            classifications[idx] = 1

    return classifications


def remove_duplicate_points(centerline, eps=0.05, min_samples=2):
    """
    Use DBSCAN clustering to remove duplicate or very close centerline points.
    :param centerline: np.array of centerline points
    :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood (cluster)
    :param min_samples: The number of points required to form a dense region (a cluster)
    :return: np.array of refined centerline points
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centerline)
    unique_points = []
    for cluster_id in set(clustering.labels_):
        if cluster_id != -1:  # Ignore noise points
            cluster_points = centerline[clustering.labels_ == cluster_id]
            # Use the mean of the cluster points as the representative point
            cluster_center = np.mean(cluster_points, axis=0)
            unique_points.append(cluster_center)
    return np.array(unique_points)


class Pointnet2dataset(Dataset):
    def __init__(self, dataset_path, file_names, num_points=1024, augment=True, split='train'):
        self.directory_path = dataset_path
        self.file_names = file_names
        self.num_points = num_points

        self.augment = augment
        self.split = split

        self.point_clouds = []
        self.labels = []
        # Split data into training, validation, and testing sets
        train_files, val_test_files = train_test_split(file_names, test_size=0.4, random_state=42)
        val_files, test_files = train_test_split(val_test_files, test_size=0.5, random_state=42)

        if self.split == 'train':
            self.file_names = train_files
        elif self.split == 'val':
            self.file_names = val_files
        elif self.split == 'test':
            self.file_names = test_files
        self.load_data()

    dataset_path = '2023_RCSE_Centerline'

    def load_data(self):
        for file_name in self.file_names:
            if file_name.endswith('.obj'):
                obj_file_path = os.path.join(self.directory_path, file_name)
            else:
                # If the file_name doesn't end with '.obj', skip or handle appropriately
                print(f"Skipping file {file_name} as it is not an OBJ file.")
                continue  # Skip this file and continue with the next one

            point_cloud = get_point_cloud(obj_file_path, self.num_points)
            point_cloud = pc_normalize(point_cloud)

            centerline_file_path = obj_file_path.replace('.obj', '_centerline.dat')
            centerline_points = load_centerline(centerline_file_path)
            centerline_points = remove_duplicate_points(centerline_points)
            centerline_points = pc_normalize(centerline_points)

            labels = classify_point(point_cloud, centerline_points)

            self.point_clouds.append(point_cloud)
            self.labels.append(labels)

    def apply_augmentations(self, point_cloud):
        if self.augment and self.split == 'train':
            point_cloud = random_scale(point_cloud)
            point_cloud = random_jitter(point_cloud)
        return point_cloud

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        labels = self.labels[idx]

        if self.split == 'train':
            # Normalize the original point cloud
            point_cloud_original = pc_normalize(point_cloud)
            point_cloud_tensor_original = torch.tensor(point_cloud_original, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            # Return the augmented point cloud
            point_cloud_augmented = self.apply_augmentations(point_cloud)
            point_cloud_augmented = pc_normalize(point_cloud_augmented)
            point_cloud_tensor_augmented = torch.tensor(point_cloud_augmented, dtype=torch.float32)

            # Concatenate the original and augmented data
            combined_point_cloud = torch.cat((point_cloud_tensor_original, point_cloud_tensor_augmented), dim=0)
            combined_labels = torch.cat((labels_tensor, labels_tensor), dim=0)

            # Debugging: Print shapes of the concatenated point cloud and labels
            print(f"Combined Point Cloud Shape: {combined_point_cloud.shape}")
            print(f"Combined Labels Shape: {combined_labels.shape}")

            # Return the concatenated data with transposed point cloud
            transposed_combined_point_cloud = combined_point_cloud.transpose(0, 1)

            # Debugging: Print shape after transposition
            print(f"Transposed Combined Point Cloud Shape: {transposed_combined_point_cloud.shape}")

            return transposed_combined_point_cloud, combined_labels

        else:
            # For validation and testing, only use the original data
            point_cloud = pc_normalize(point_cloud)
            point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            # Debugging: Print shapes for validation and testing
            print(f"Validation/Test Point Cloud Shape: {point_cloud_tensor.shape}")
            print(f"Validation/Test Labels Shape: {labels_tensor.shape}")

            return point_cloud_tensor.transpose(0, 1), labels_tensor

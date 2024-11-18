import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import trimesh
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch.utils.data import Dataset, DataLoader

from data_utils.MeshCenterline import load_centerline
from models.pointnet_utils import PointNetEncoder


class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        pred = pred.permute(0, 2, 1)
        return F.nll_loss(pred, target)


# Point Cloud Utilities
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc /= m
    return pc


def classify_point(point_cloud, centerline_points, threshold=0.05):
    tree = cKDTree(point_cloud)
    classifications = np.zeros(len(point_cloud))
    for point in centerline_points:
        distance, idx = tree.query(point)
        if distance <= threshold:
            classifications[idx] = 1
    return classifications


def remove_duplicate_points(centerline, eps=0.05, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centerline)
    unique_points = [np.mean(centerline[clustering.labels_ == cluster], axis=0)
                     for cluster in set(clustering.labels_) if cluster != -1]
    return np.array(unique_points)


# Custom Dataset
class PointNetDataset(Dataset):
    def __init__(self, dataset_path, file_names, num_points=1024):
        self.dataset_path = dataset_path
        self.file_names = file_names
        self.num_points = num_points
        self.point_clouds, self.labels = self.load_data()

    def load_data(self):
        point_clouds, labels = [], []
        for file_name in self.file_names:
            file_path = os.path.join(self.dataset_path, file_name)
            if not file_path.endswith('.obj'):
                continue
            mesh = trimesh.load(file_path)
            point_cloud = pc_normalize(mesh.sample(self.num_points))
            centerline_points = pc_normalize(
                remove_duplicate_points(load_centerline(file_path.replace('.obj', '_centerline.dat'))))
            labels.append(classify_point(point_cloud, centerline_points))
            point_clouds.append(point_cloud)
        return point_clouds, labels

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = torch.tensor(self.point_clouds[idx], dtype=torch.float32).transpose(0, 1)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return point_cloud, labels


# Visualization
def visualize_mesh_and_predictions(mesh, centerline_points, point_cloud, classifications, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    random_indices = np.random.choice(len(mesh.vertices), 1000, replace=False)
    vertices = mesh.vertices[random_indices]
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="blue", label="Random vertices")
    ax.scatter(centerline_points[:, 0], centerline_points[:, 1], centerline_points[:, 2], c="red", label="Centerline")
    ax.scatter(point_cloud[classifications == 1, 0], point_cloud[classifications == 1, 1],
               point_cloud[classifications == 1, 2], c="green", label="Classified Points")
    plt.title(title)
    plt.legend()
    plt.show()


# Loading and splitting dataset
dataset_path = '2023_RCSE_Centerline'
file_list = os.listdir(dataset_path)
obj_files = [file for file in file_list if file.endswith('.obj')]
train_files, test_files = obj_files[:40], obj_files[40:]

# Training and Testing Datasets
train_dataset = PointNetDataset(dataset_path, train_files)
test_dataset = PointNetDataset(dataset_path, test_files)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Model, Loss, and Optimizer
num_classes = 2
model = get_model(num_classes)
criterion = get_loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

# Evaluation
model.eval()
true_labels, predicted_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs, _ = model(inputs)
        predicted = (outputs > 0).squeeze().int()
        true_labels.extend(labels.view(-1).tolist())
        predicted_labels.extend(predicted.view(-1).tolist())

# Metrics Calculation
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f'Accuracy: {accuracy}, Precision: {precision}, F1 Score: {f1}')

# Visualization Before Training
file_path = '2023_RCSE_Centerline/AC_20150820_left.obj'  # replace with actual path
mesh = trimesh.load(file_path)
point_cloud = pc_normalize(mesh.sample(2048))
centerline_points = pc_normalize(remove_duplicate_points(load_centerline(file_path.replace('.obj', '_centerline.dat'))))
classifications = classify_point(point_cloud, centerline_points)
visualize_mesh_and_predictions(mesh, centerline_points, point_cloud, classifications, "Before Training")

# Visualization After Training
new_file_path = '2023_RCSE_Centerline/MR_17122020.obj'  # replace with actual path
mesh = trimesh.load(new_file_path)
new_point_cloud = pc_normalize(mesh.sample(2048)).reshape(1, 3, -1)
outputs, _ = model(torch.tensor(new_point_cloud, dtype=torch.float32))
predicted = (outputs > 0).squeeze().numpy().astype(int)
visualize_mesh_and_predictions(mesh, centerline_points, new_point_cloud[0].T, predicted, "After Training")

import argparse
import os

import pyvista as pv
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

from data_utils.MeshCenterline import Pointnet2dataset
from models.pointnet2_sem_seg import get_model, get_loss


def parse_args():
    parser = argparse.ArgumentParser('PointNet++ Training and Evaluation')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet2_cls_ssg]')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs in training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate in training')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points in point cloud')
    parser.add_argument('--optimizer', default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    return parser.parse_args()


def evaluate_model(model, loader, device, criterion):
    model.eval()
    losses = []
    all_labels = []
    all_preds = []
    point_clouds = []  # To store point clouds for visualization
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs, _ = model(data)  # assuming model returns (outputs, _) where _ is unused
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds = outputs.argmax(dim=2).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            point_clouds.extend(data.cpu().numpy())  # Store point clouds for visualization
    # Calculate average loss
    avg_loss = sum(losses) / len(losses)
    # Flatten lists for metrics calculations
    all_labels = [label for sublist in all_labels for label in sublist]
    all_preds = [pred for sublist in all_preds for pred in sublist]
    # Calculate precision, recall, and accuracy
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, precision, recall, accuracy


def visualize_point_cloud_with_labels(point_cloud, true_labels, pred_labels):
    point_cloud = point_cloud.T  # Transpose to (X, 3)
    plotter = pv.Plotter()

    # Convert to pyvista-compatible data
    lidar_points = pv.PolyData(point_cloud)
    lidar_points['true_labels'] = true_labels
    lidar_points['pred_labels'] = pred_labels

    # True labels visualization
    plotter.add_mesh(lidar_points, scalars='true_labels', cmap='viridis', point_size=5, render_points_as_spheres=True)
    plotter.add_text("True Labels", font_size=10, position='upper_left')
    plotter.show()

    # Predicted labels visualization
    plotter.add_mesh(lidar_points, scalars='pred_labels', cmap='plasma', point_size=5, render_points_as_spheres=True)
    plotter.add_text("Predicted Labels", font_size=10, position='upper_left')
    plotter.show()


def visualize_predictions(test_loader, model, device):
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs, _ = model(data)
            preds = outputs.argmax(dim=2).cpu().numpy()
            data = data.cpu().numpy()
            labels = labels.cpu().numpy()

            # Visualize a batch
            for i in range(len(data)):
                visualize_point_cloud_with_labels(data[i], labels[i], preds[i])
                break  # Visualize only the first sample in the batch
            break  # Visualize only one batch


def main(args):
    dataset_path = '2023_RCSE_Centerline'
    file_names = os.listdir(dataset_path)

    train_dataset = Pointnet2dataset(dataset_path, file_names, num_points=args.num_points, augment=True, split='train')
    val_dataset = Pointnet2dataset(dataset_path, file_names, num_points=args.num_points, augment=False, split='val')
    test_dataset = Pointnet2dataset(dataset_path, file_names, num_points=args.num_points, augment=False, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    model = get_model(num_classes=2).to(device)
    print("Model successfully initialized")
    criterion = get_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)

    # Training loop
    for i, epoch in enumerate(range(args.epoch)):
        model.train()
        for j, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(data)
            print("OUTPUTS SHAPE", outputs.shape)
            print("LABELS SHAPE", labels.shape)
            loss = criterion(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()
            if j == 2:
                break
        if i == 2:
            break

        train_loss, train_prec, train_recall, train_acc = evaluate_model(model, train_loader, device, criterion)
        val_loss, val_prec, val_recall, val_acc = evaluate_model(model, val_loader, device, criterion)
        print(
            f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Precision: {train_prec:.4f}, Recall: {train_recall:.4f}, Accuracy: {train_acc:.4f}')
        print(
            f'Validation Loss: {val_loss:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, Accuracy: {val_acc:.4f}')

    # Testing evaluation
    test_loss, test_prec, test_recall, test_acc = evaluate_model(model, test_loader, device, criterion)
    print(
        f'Test Loss: {test_loss:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, Accuracy: {test_acc:.4f}')

    # Visualize the model on the test dataset
    visualize_predictions(test_loader, model, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)

# import argparse
# import os
#
# import pyvista as pv
# import torch
# from sklearn.metrics import precision_score, recall_score, accuracy_score
# from torch.utils.data import DataLoader
#
# from data_utils.MeshCenterline import Pointnet2dataset
# from models.pointnet2_sem_seg import get_model, get_loss
#
#
# def parse_args():
#     parser = argparse.ArgumentParser('PointNet++ Training and Evaluation')
#     parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
#     parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
#     parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
#     parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet2_cls_ssg]')
#     parser.add_argument('--epoch', type=int, default=100, help='number of epochs in training')
#     parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate in training')
#     parser.add_argument('--num_points', type=int, default=1024, help='number of points in point cloud')
#     parser.add_argument('--optimizer', default='Adam', help='optimizer for training')
#     parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
#     parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
#     parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
#     return parser.parse_args()


# def visualize_lidar_and_centerline_pyvista(lidar_points, centerline_points):
#     plotter = pv.Plotter()
#
#     # Add LiDAR points
#     lidar_cloud = pv.PolyData(lidar_points)
#     plotter.add_mesh(lidar_cloud, color='blue', point_size=3, render_points_as_spheres=True)
#
#     # Add centerline points
#     centerline_cloud = pv.PolyData(centerline_points)
#     plotter.add_mesh(centerline_cloud, color='red', point_size=5, render_points_as_spheres=True)
#
#     # Show the interactive plot
#     plotter.show()


# def evaluate_model(model, loader, device, criterion):
#     """
#     Evaluate the model using the provided data loader.
#
#     Args:
#         model (nn.Module): The model to evaluate.
#         loader (DataLoader): The DataLoader providing data and labels.
#         device (torch.device): The device to which tensors should be moved.
#         criterion (nn.Module): The loss function (e.g., nn.NLLLoss()).
#
#     Returns:
#         avg_loss (float): The average loss across the dataset.
#         precision (float): The precision of the predictions.
#         recall (float): The recall of the predictions.
#         accuracy (float): The accuracy of the predictions.
#         outputs_list (list): A list of dictionaries containing inputs, labels, and predictions.
#     """
#     model.eval()
#     losses = []
#     outputs_list = []  # To store outputs for visualization
#
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for data, labels in loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs, _ = model(data)  # Assuming model returns (outputs, _) where _ is unused
#
#             # Compute loss
#             loss = criterion(outputs, labels)
#             losses.append(loss.item())
#
#             # Get predictions
#             preds = outputs.argmax(dim=2).detach().cpu().numpy()
#             labels_np = labels.detach().cpu().numpy()
#             data_np = data.detach().cpu().numpy()
#
#             # Save each sample for visualization
#             for i in range(len(data_np)):
#                 outputs_list.append({
#                     'input': data_np[i],  # Save the input sample
#                     'label': labels_np[i],  # True label
#                     'prediction': preds[i]  # Model prediction
#                 })
#
#             # Extend for metrics calculation
#             all_preds.extend(preds)
#             all_labels.extend(labels_np)
#
#     # Calculate average loss
#     avg_loss = sum(losses) / len(losses)
#
#     # Flatten labels and predictions for metrics
#     all_labels = [label for sublist in all_labels for label in sublist]
#     all_preds = [pred for sublist in all_preds for pred in sublist]
#
#     # Calculate precision, recall, and accuracy
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     accuracy = accuracy_score(all_labels, all_preds)
#
#     return avg_loss, precision, recall, accuracy, outputs_list
#
# def evaluate_model(model, loader, device, criterion):
#     model.eval()
#     losses = []
#     all_labels = []
#     all_preds = []
#     point_clouds = []  # To store point clouds for visualization
#     with torch.no_grad():
#         for data, labels in loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs, _ = model(data)  # assuming model returns (outputs, _) where _ is unused
#             loss = criterion(outputs, labels)
#             losses.append(loss.item())
#             preds = outputs.argmax(dim=2).detach().cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels.detach().cpu().numpy())
#             point_clouds.extend(data.cpu().numpy())  # Store point clouds for visualization
#     # Calculate average loss
#     avg_loss = sum(losses) / len(losses)
#     # Flatten lists for metrics calculations
#     all_labels = [label for sublist in all_labels for label in sublist]
#     all_preds = [pred for sublist in all_preds for pred in sublist]
#     # Calculate precision, recall, and accuracy
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     accuracy = accuracy_score(all_labels, all_preds)
#     return avg_loss, precision, recall, accuracy
#
#
# def main(args):
#     dataset_path = '2023_RCSE_Centerline'
#     file_names = os.listdir(dataset_path)
#
#     train_dataset = Pointnet2dataset(dataset_path, file_names, num_points=args.num_points, augment=True, split='train')
#     val_dataset = Pointnet2dataset(dataset_path, file_names, num_points=args.num_points, augment=False, split='val')
#     test_dataset = Pointnet2dataset(dataset_path, file_names, num_points=args.num_points, augment=False, split='test')
#
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
#
#     device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
#     model = get_model(num_classes=2).to(device)
#     print("model successfully initialized")
#     print(model)
#     criterion = get_loss().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
#
#     # Training loop
#     for i, epoch in enumerate(range(args.epoch)):
#         model.train()
#         for j, (data, labels) in enumerate(train_loader):
#             data, labels = data.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs, _ = model(data)
#             print("OUTPUTS SHAPE", outputs.shape)
#             print("LABELS SHAPE", labels.shape)
#             loss = criterion(outputs, labels)
#             print(loss)
#             loss.backward()
#             optimizer.step()
#             if j == 2:
#                 break
#         if i == 2:
#             break
#
#         train_loss, train_prec, train_recall, train_acc = evaluate_model(model, train_loader, device, criterion)
#         val_loss, val_prec, val_recall, val_acc = evaluate_model(model, val_loader, device, criterion)
#         print(
#             f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Precision: {train_prec:.4f}, Recall: {train_recall:.4f}, Accuracy: {train_acc:.4f}')
#         print(
#             f'Validation Loss: {val_loss:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, Accuracy: {val_acc:.4f}')
#
#     # Testing evaluation
#     test_loss, test_prec, test_recall, test_acc = evaluate_model(model, test_loader, device, criterion)
#     print(
#         f'Test Loss: {test_loss:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, Accuracy: {test_acc:.4f}')
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     main(args)

#
# # Training loop
# for i, epoch in enumerate(range(args.epoch)):
#     model.train()
#     for j, (data, labels) in enumerate(train_loader):
#         data, labels = data.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs, _ = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#     train_loss, train_prec, train_recall, train_acc = evaluate_model(model, train_loader, device, criterion)
#     val_loss, val_prec, val_recall, val_acc = evaluate_model(model, val_loader, device, criterion)
#     print(
#         f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Precision: {train_prec:.4f}, Recall: {train_recall:.4f}, Accuracy: {train_acc:.4f}')
#     print(
#         f'Validation Loss: {val_loss:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, Accuracy: {val_acc:.4f}')
#
# #Saving model
#
# # Testing evaluation
# test_loss, test_prec, test_recall, test_acc, outputs_list = evaluate_model(
#     model, test_loader, device, criterion)
# print(
#     f'Test Loss: {test_loss:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, Accuracy: {test_acc:.4f}')
#
# # Visualize a single result
# single_result = outputs_list[0]  # Select the first result for visualization
# print("\nVisualizing a Single Result:")
# print(f"Input: {single_result['input']}")
# print(f"True Label: {single_result['label']}")
# print(f"Prediction: {single_result['prediction']}")
#
# # Visualize using PyVista
# visualize_lidar_and_centerline_pyvista(
#     lidar_points=single_result['input'],
#     centerline_points=single_result['prediction']
# )

# if __name__ == '__main__':
#     args = parse_args()
#     main(args)

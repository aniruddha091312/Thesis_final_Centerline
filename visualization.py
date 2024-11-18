import argparse

import torch
from networkx.algorithms.distance_measures import center
from numpy.ma.core import shape
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

from data_utils.MeshCenterline_infer import Pointnet2dataset
from models.pointnet2_sem_seg import get_model, get_loss
import numpy as np
import pyvista as pv

def evaluate_model(model, loader, device, criterion):
    """
    Evaluate the model using the provided data loader.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): The DataLoader providing data and labels.
        device (torch.device): The device to which tensors should be moved.
        criterion (nn.Module): The loss function (e.g., nn.NLLLoss()).

    Returns:
        avg_loss (float): The average loss across the dataset.
        precision (float): The precision of the predictions.
        recall (float): The recall of the predictions.
        accuracy (float): The accuracy of the predictions.
        outputs_list (list): A list of dictionaries containing inputs, labels, and predictions.
    """
    model.eval()
    losses = []
    outputs_list = []  # To store outputs for visualization

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs, _ = model(data)  # Assuming model returns (outputs, _) where _ is unused

            # Compute loss
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Get predictions
            preds = outputs.argmax(dim=2).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            data_np = data.detach().cpu().numpy()

            # Save each sample for visualization
            for i in range(len(data_np)):
                outputs_list.append({
                    'input': data_np[i],  # Save the input sample
                    'label': labels_np[i],  # True label
                    'prediction': preds[i]  # Model prediction
                })

            # Extend for metrics calculation
            all_preds.extend(preds)
            all_labels.extend(labels_np)

    # Calculate average loss
    avg_loss = sum(losses) / len(losses)

    # Flatten labels and predictions for metrics
    all_labels = [label for sublist in all_labels for label in sublist]
    all_preds = [pred for sublist in all_preds for pred in sublist]

    # Calculate precision, recall, and accuracy
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, precision, recall, accuracy, outputs_list


def vis(dir_path: str) -> None:
    dataset = Pointnet2dataset(dataset_path="", file_names="", num_points=1024, augment=False)
    print(dataset)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data in enumerate(test_loader):
        pass

    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    model = get_model(num_classes=2).to(device)
    #model.load_state_dict(torch.load(dir_path))
    print("model successfully initialized")
    print(model)
    criterion = get_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.008)

    _, _, _, _, outputs_list = evaluate_model(model, test_loader, device, criterion)

    print(outputs_list[0]['input'].shape, outputs_list[0]['label'].shape, outputs_list[0]['prediction'].shape)
    print(outputs_list)

    input_list = np.array(outputs_list[0]['input'])
    mod_input_list = input_list.T


    centerline =[]
    non_centerline=[]
    for i, p in enumerate(outputs_list[0]['prediction']):
        if p==1:
            centerline.append(mod_input_list[i])
        else:
            non_centerline.append(mod_input_list[i])

    if len(non_centerline) == 0:
        non_centerline.append([0, 0, 0])
    elif len(centerline) == 0:
        centerline.append([0, 0, 0])
    else:
        pass

    visualize_lidar_and_centerline_pyvista(non_centerline, centerline)


def visualize_lidar_and_centerline_pyvista(lidar_points, centerline_points):
    plotter = pv.Plotter()

    # Add LiDAR points
    lidar_cloud = pv.PolyData(lidar_points)
    plotter.add_mesh(lidar_cloud, color='blue', point_size=3, render_points_as_spheres=True)

    # Add centerline points
    centerline_cloud = pv.PolyData(centerline_points)
    plotter.add_mesh(centerline_cloud, color='red', point_size=5, render_points_as_spheres=True)

    # Show the interactive plot
    plotter.show()

def main(dir_path):
    """
    Main function to process the file.

    Args:
        filepath (str): The path to the file provided as an argument.
    """
    try:
        # Example logic to demonstrate handling of the file
        print(f"Processing file: {dir_path}")
        # Add your file processing logic here

        vis(dir_path)


    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Parse, process and visualize a file.")

    # Add argument for file path
    parser.add_argument(
        '--filepath',
        type=str,
        required=True,
        help="The path to the file to be processed."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed file path
    main(args.filepath)

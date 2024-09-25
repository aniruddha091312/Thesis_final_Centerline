import argparse
import os

import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader

from KPconv.models.architecture import KPFCNN
from KPconv.utils.config import Config
from data_utils.KPconv_dataset import Pointnet2dataset, CustomBatch


def parse_args():
    parser = argparse.ArgumentParser('KPCONV Training and Evaluation')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs in training')
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

    with torch.no_grad():
        for batch in loader:
            batch = CustomBatch(batch).to(device)  # Wrap the batch in CustomBatch and move to device

            outputs = model(batch)  # Pass the CustomBatch instance to the model

            # Reshape outputs and labels for evaluation
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs to [batch_size * num_points, num_classes]
            labels = batch.labels.view(-1)  # Flatten labels to [batch_size * num_points]

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = sum(losses) / len(losses)
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, precision, recall, accuracy


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

    config = Config()
    config.num_classes = 2  # Assuming a binary segmentation task
    lbl_values = [0, 1]
    ign_lbls = []

    model = KPFCNN(config, lbl_values, ign_lbls).to(device)
    print("KPConv model successfully initialized")
    print(model)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)

    # Training loop
    for epoch in range(args.epoch):
        model.train()
        for batch in train_loader:
            batch = CustomBatch(batch).to(device)  # Wrap the batch in CustomBatch and move to device
            optimizer.zero_grad()

            outputs = model(batch)  # Model output should be [batch_size, num_points, num_classes]

            # Reshape outputs and labels for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs to [batch_size * num_points, num_classes]
            labels = batch.labels.view(-1)  # Flatten labels to [batch_size * num_points]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_prec, train_recall, train_acc = evaluate_model(model, train_loader, device, criterion)
        val_loss, val_prec, val_recall, val_acc = evaluate_model(model, val_loader, device, criterion)
        print(
            f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Precision: {train_prec:.4f}, Recall: {train_recall:.4f}, Accuracy: {train_acc:.4f}')
        print(
            f'Validation Loss: {val_loss:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, Accuracy: {val_acc:.4f}')

    test_loss, test_prec, test_recall, test_acc = evaluate_model(model, test_loader, device, criterion)
    print(
        f'Test Loss: {test_loss:.4f}, Precision: {test_prec:.4f}, Recall: {test_recall:.4f}, Accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

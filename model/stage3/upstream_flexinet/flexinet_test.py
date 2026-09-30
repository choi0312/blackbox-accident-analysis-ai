import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from dataset.nuimages import NuImagesDataset
from dataset.kitti import KITTIDataset
from model.FlexiNet import FlexiNet
from torch.utils.tensorboard import SummaryWriter
import sys

def set_seed(seed_value):
    """
    Set seed for reproducibility across various libraries and frameworks.
    
    Args:
        seed_value (int): The seed value to set.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)          # Set seed for the current GPU
    torch.cuda.manual_seed_all(seed_value)      # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True    # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False       # Disable benchmarking for reproducibility


def test(args, model, dataloader_test):
    """
    Perform testing on the test dataset.
    Use RMSE for 'L2' loss type and MAE for 'L1' loss type.
    
    Args:
        args (Namespace): Parsed command-line arguments.
        model (nn.Module): The neural network model.
        dataloader_test (DataLoader): DataLoader for the test dataset.
    
    Returns:
        float: Average test loss.
    """
    model.eval()  # Set model to evaluation mode
    running_test_loss = 0.0
    eps = 1e-6  # Small constant for numerical stability
    
    # Select the appropriate loss function and behavior based on the criterion type
    if args.loss_type == 'L1':
        criterion = nn.L1Loss()  # Mean Absolute Error
    elif args.loss_type == 'L2':
        criterion = nn.MSELoss()  # Mean Squared Error 
    else:
        raise ValueError("Invalid loss type specified. Choose either 'L1' or 'L2'.")

    with torch.no_grad():  # Disable gradient computation
        for data in tqdm(dataloader_test, desc="Testing", unit="batch"):
            img_batch, speed = data
            img_batch, speed = img_batch.to(args.device), speed.to(args.device)
            predicted_speed = model(img_batch)
            
            if args.loss_type == 'L1':
                # Directly calculate MAE
                loss = criterion(predicted_speed, speed.unsqueeze(1))
            elif args.loss_type == 'L2':
                # Calculate RMSE
                mse_loss = criterion(predicted_speed, speed.unsqueeze(1))
                loss = torch.sqrt(mse_loss + eps)  # RMSE computation
            
            running_test_loss += loss.item()

    # Calculate average test loss
    average_test_loss = running_test_loss / len(dataloader_test)
    return average_test_loss


def load_model(model, checkpoint_path, device):
    """
    Load the model checkpoint.

    Args:
        model (nn.Module): The model instance.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: Model loaded with the checkpoint weights.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        # If 'state_dict' exists, load it
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        # If the checkpoint is a state_dict itself
        model.load_state_dict(checkpoint)
    else:
        raise KeyError("'state_dict' key not found in checkpoint file, and checkpoint is not a valid state_dict.")
    
    return model



def main(params):
    """
    Main function to execute testing.
    
    Args:
        params (list): List of command-line arguments.
    """
    # Set seed for reproducibility
    set_seed(0)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Testing Pipeline for FlexiNet")
    parser.add_argument('--data', type=str, required=True, help='Path of test data')
    parser.add_argument('--dataset', type=str, default="Kitti", help='Dataset to use: Kitti, nuImages')
    parser.add_argument('--crop_height', type=int, default=64, help='Height of cropped input image to network')
    parser.add_argument('--crop_width', type=int, default=64, help='Width of cropped input image to network')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--cuda', type=str, default='0', help='GPU id(s) to use for testing')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Whether to use GPU for testing')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--save_model_path', type=str, required=True, help='Path to save the model')
    parser.add_argument('--loss_type', type=str, choices=['L1', 'L2'], required=True, help='Loss function type: L1 or L2')

    args = parser.parse_args(params)

    # Set CUDA devices before any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {args.device}")

    # Load the appropriate dataset
    dataset_name = args.dataset.lower()
    if dataset_name == 'kitti':
        dataset_test = KITTIDataset(
            dataroot=args.data,
            split='valid',
            img_size=(args.crop_height, args.crop_width),
        )
    elif dataset_name == 'nuimages':
        dataset_test = NuImagesDataset(
            dataroot=args.data,
            version='v1.0-test',
            img_size=(args.crop_height, args.crop_width),
            subset=False
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from 'Kitti', 'nuImages'.")

    # Create DataLoader for test dataset
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # Initialize the model
    model = FlexiNet(input_channels=1, num_classes=1)

    # Use multiple GPUs if available and specified
    if torch.cuda.device_count() > 1 and args.use_gpu:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for testing.")

    model = model.to(args.device)  # Move model to the specified device
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    # Load pretrained model
    model = load_model(model, args.checkpoint_path, args.device)

    # Perform testing
    test_loss = test(args, model, dataloader_test)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    """
    Entry point of the testing script. Parses command-line arguments and starts the testing process.
    
    Example usage:
        python evaluate.py --data ./data/kitti --dataset Kitti --checkpoint_path ./log/model.pth --save_model_path ./log --loss_type L1
    """
    main(sys.argv[1:])

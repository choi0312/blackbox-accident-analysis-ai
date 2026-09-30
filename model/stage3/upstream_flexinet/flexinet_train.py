import os
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
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


def val(args, model, dataloader_val):
    """
    Perform validation on the validation dataset.
    
    Args:
        args (Namespace): Parsed command-line arguments.
        model (nn.Module): The neural network model.
        dataloader_val (DataLoader): DataLoader for the validation dataset.
    
    Returns:
        float: Average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0

    # Select loss function based on loss_type
    if args.loss_type.upper() == 'L1':
        criterion = nn.L1Loss()
    elif args.loss_type.upper() == 'L2':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}. Choose 'L1' or 'L2'.")

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, data in enumerate(tqdm(dataloader_val, desc="Validating", unit="batch")):
            img_batch, speed = data
            img_batch = img_batch.to(args.device).float()  # Ensure float type
            speed = speed.to(args.device).float()          # Convert to float

   

            predicted_speed = model(img_batch)
            # Compute loss; ensure target has the correct shape
            val_loss = criterion(predicted_speed, speed.unsqueeze(1))

            running_val_loss += val_loss.item()

    # Calculate average validation loss
    average_val_loss = running_val_loss / len(dataloader_val)
    return average_val_loss


def train(args, model, optimizer, scheduler, dataloader_train, dataloader_val):
    """
    Train the model with checkpointing and validation.
    
    Args:
        args (Namespace): Parsed command-line arguments.
        model (nn.Module): The neural network model.
        optimizer (Optimizer): Optimizer for training.
        scheduler (Scheduler): Learning rate scheduler.
        dataloader_train (DataLoader): DataLoader for the training dataset.
        dataloader_val (DataLoader): DataLoader for the validation dataset.
    """
    model.to(args.device)  # Move model to the specified device

    # Ensure all model parameters are float
    model = model.float()

    # Select loss function based on loss_type
    if args.loss_type.upper() == 'L1':
        criterion = nn.L1Loss()
    elif args.loss_type.upper() == 'L2':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}. Choose 'L1' or 'L2'.")

    # Modify save_model_path to include loss_type
    save_model_path = os.path.join(args.save_model_path, args.loss_type.upper())
    os.makedirs(save_model_path, exist_ok=True)  # Create directory to save models

    # Initialize TensorBoard writer with log directory based on loss_type
    writer = SummaryWriter(log_dir=save_model_path)

    best_val_loss = float('inf')  # Initialize best validation loss
    start_epoch = 0  # Initialize starting epoch

    # Load checkpoint if specified
    if args.checkpoint_path:
        if os.path.isfile(args.checkpoint_path):
            start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)
            print(f"Loaded checkpoint from {args.checkpoint_path}, starting at epoch {start_epoch + 1}")
        else:
            print(f"Checkpoint path '{args.checkpoint_path}' is invalid. Starting from scratch.")

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Iterate over epochs
    for epoch in range(start_epoch, args.num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over batches
        for batch_idx, data in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch")):
            img_batch, speed = data

            # Ensure tensors are float and moved to device
            img_batch = img_batch.to(args.device).float()
            speed = speed.to(args.device).float()

            optimizer.zero_grad()  # Zero the gradients

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                predicted_speed = model(img_batch)  # Forward pass
                loss = criterion(predicted_speed, speed.unsqueeze(1))  # Compute loss


            if args.use_amp:
                scaler.scale(loss).backward()  # Backward pass with scaling
                scaler.step(optimizer)         # Update weights
                scaler.update()                # Update scaler
            else:
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

            running_loss += loss.item()

        # Calculate average training loss for the epoch
        train_loss = running_loss / len(dataloader_train)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Train Loss: {train_loss:.4f}")
        writer.add_scalar('Loss/Train', train_loss, epoch)  # Log training loss

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

        scheduler.step(train_loss)  # Update learning rate based on scheduler

        # Save checkpoint at specified intervals
        if (epoch + 1) % args.checkpoint_step == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            # Extract dataset name in lowercase for filename consistency
            dataset_name = args.dataset.lower()
            checkpoint_filename = f'checkpoint_epoch_{epoch + 1}_{dataset_name}_{args.loss_type.upper()}_best.pth'
            checkpoint_path = os.path.join(save_model_path, checkpoint_filename)
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Perform validation at specified intervals
        if (epoch + 1) % args.validation_step == 0:
            val_loss = val(args, model, dataloader_val)
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Validation Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/Validation', val_loss, epoch)  # Log validation loss

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Extract dataset name in lowercase for filename consistency
                dataset_name = args.dataset.lower()
                best_model_filename = f'checkpoint_epoch_{epoch + 1}_{dataset_name}_{args.loss_type.upper()}_best.pth'
                best_model_path = os.path.join(save_model_path, best_model_filename)
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model: {best_model_path}")

    writer.close()  # Close the TensorBoard writer


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer states from a checkpoint.
    
    Args:
        model (nn.Module): The neural network model.
        optimizer (Optimizer): Optimizer for training.
        checkpoint_path (str): Path to the checkpoint file.
    
    Returns:
        int: The epoch to resume training from.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load checkpoint
    if isinstance(model, nn.DataParallel):
        # Adjust keys if model was saved using DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)  # Load model state
    optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer state
    start_epoch = checkpoint['epoch'] + 1  # Set starting epoch
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch


def main(params):
    """
    Main function to execute training and validation.
    
    Args:
        params (list): List of command-line arguments.
    """
    # Set seed for reproducibility
    set_seed(0)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training Pipeline for FlexiNet")
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs to train for')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation (epochs)')
    parser.add_argument('--data', type=str, required=True, help='Path of training data')
    parser.add_argument('--dataset', type=str, default="Kitti", help='Dataset to use: Kitti, nuImages, or modelTest')
    parser.add_argument('--crop_height', type=int, default=64, help='Height of cropped input image to network')
    parser.add_argument('--crop_width', type=int, default=64, help='Width of cropped input image to network')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of object classes')
    parser.add_argument('--cuda', type=str, default='0', help='GPU id(s) to use for training, e.g., "0" or "0,1"')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Whether to use GPU for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--save_model_path', type=str, required=True, help='Path to save the model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--loss_type', type=str, choices=['L1', 'L2'], default='L1', help='Type of loss to use: L1 or L2')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use Automatic Mixed Precision for training')

    args = parser.parse_args(params)

    # Set CUDA devices before any CUDA operations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {args.device}")

    # Load the appropriate dataset
    dataset_name = args.dataset.lower()

    if dataset_name == 'kitti':
        dataset_train = KITTIDataset(
            dataroot=args.data,
            split='train',
            img_size=(args.crop_height, args.crop_width),
        )
        dataset_val = KITTIDataset(
            dataroot=args.data,
            split='valid',
            img_size=(args.crop_height, args.crop_width),
        )
    elif dataset_name == 'nuimages':
        dataset_train = NuImagesDataset(
            dataroot=args.data,
            version='v1.0-train',
            img_size=(args.crop_height, args.crop_width),
            subset=False
        )
        dataset_val = NuImagesDataset(
            dataroot=args.data,
            version='v1.0-val',
            img_size=(args.crop_height, args.crop_width),
            subset=False
        )
    elif dataset_name == 'modeltest':
        # Debug mode: Test the model architecture with dummy data
        model = FlexiNet(input_channels=1, num_classes=1)
        input_tensor = torch.randn(4, 13, 1, 64, 64).to(args.device).float()  # Example input tensor
        model.to(args.device)
        model = model.float()  # Ensure model parameters are float
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        print("Model Test Output:", output)
        return
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from 'Kitti', 'nuImages', or 'modelTest'.")

    # Create DataLoaders for training and validation datasets
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # Initialize the model
    model = FlexiNet(input_channels=1, num_classes=1)

    # Use multiple GPUs if available and specified
    if torch.cuda.device_count() > 1 and args.use_gpu:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for training.")

    model = model.to(args.device)  # Move model to the specified device

    # Ensure all model parameters are float
    model = model.float()

    print('Models parameters', sum(p.numel() for p in model.parameters()))

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=10,
        factor=0.1,
    )

    # Load pretrained model if provided
    if args.pretrained_model_path:
        if os.path.isfile(args.pretrained_model_path):
            print(f'Loading pretrained model from {args.pretrained_model_path}...')
            pretrained_state = torch.load(args.pretrained_model_path, map_location=args.device)
            if isinstance(model, nn.DataParallel):
                # If using DataParallel, load state_dict to module
                model.module.load_state_dict(pretrained_state)
            else:
                model.load_state_dict(pretrained_state)
            print('Pretrained model loaded successfully!')
        else:
            print(f"Pretrained model path '{args.pretrained_model_path}' is invalid. Continuing without loading.")

    # Start training
    train(args, model, optimizer, scheduler, dataloader_train, dataloader_val)


if __name__ == '__main__':
    """
    Entry point of the script. Parses command-line arguments and starts the training process.
    
    Example usage:
        python train.py --data ./data/nuscenes --dataset nuImages --save_model_path ./log/nuscenes_flexinet_model --loss_type L1 --use_amp
    """
    # Pass command-line arguments excluding the script name
    main(sys.argv[1:])

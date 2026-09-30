import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset import custom_transforms as tr


class KITTIDataset(Dataset):
    def __init__(self, dataroot, split='train', sequence_length=13, img_size=(64,64), train_mode='gray', frame_skip_interval=1):
        self.dataroot = os.path.join(dataroot, split)
        self.split = split
        self.sequence_length = sequence_length

        # Resize
        self.img_height=img_size[0] # (h,w)
        self.img_width=img_size[1] # (h,w)
        
        self.train_mode = train_mode
        self.frame_skip_interval = frame_skip_interval # 1 = 10 Hz, 5 = 2Hz

        self.samples = self._load_split_data()

        if split == 'train':
            self.transform = self.transform_tr()
        elif split == 'valid':
            self.transform = self.transform_val()
        else:  # Fallback to a default transformation
            self.transform = self.transform_ts()


    def _load_split_data(self):
        samples = []
        for drive_folder in sorted(os.listdir(self.dataroot)):
            date_part = drive_folder.split('_')[0] + '_' + drive_folder.split('_')[1] + '_' + drive_folder.split('_')[2]
            drive_path = os.path.join(self.dataroot, drive_folder, date_part, drive_folder)
            image_dir = os.path.join(drive_path, "image_03", "data")
            oxts_dir = os.path.join(drive_path, "oxts", "data")

            if os.path.isdir(image_dir) and os.path.isdir(oxts_dir):
                image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
                
                # Adjust the range to account for skipped frames
                for start_idx in range(0, len(image_files) - self.sequence_length * self.frame_skip_interval + 1, self.frame_skip_interval):
                    sequence_files = [image_files[start_idx + i * self.frame_skip_interval] for i in range(self.sequence_length)]
                    img_paths = [os.path.join(image_dir, f) for f in sequence_files]
                    imu_paths = [os.path.join(oxts_dir, f.replace('.png', '.txt')) for f in sequence_files]
                    samples.append((img_paths, imu_paths))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths, imu_paths = self.samples[idx]
        
        # Initialize a list to hold the loaded images
        images = []
        
        # Load each image in the sequence and apply transformations
        for img_path in img_paths:
            if self.train_mode =='gray':
                image = Image.open(img_path).convert('L')
            else:
                image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # Stack images along a new dimension to create a single tensor
        # Resulting tensor shape: (sequence_length, C, H, W)
        images_stacked = torch.stack(images, dim=0)
        
        # Load IMU data (forward velocity) for each frame and stack them
        vf = [np.loadtxt(imu_path)[8] for imu_path in imu_paths]
        vf_tensor = torch.tensor(vf, dtype=torch.float)
        

        # Exponential Moving Average (EMA) speed for a sequence of speeds
        # vf_tensor_est = self.calculate_final_ema_speed(vf_tensor)

        vf_tensor_est = self.weighted_avg_speed(vf_tensor)
        
        # vf_tensor_est = vf_tensor.mean()

        return images_stacked, vf_tensor_est
    
    def weighted_avg_speed(self, speeds_tensor):

        weights = torch.linspace(1, 2, steps=13)  # Increasing weights for later frames
        weighted_avg_speed = (speeds_tensor * weights).sum() / weights.sum()

        return weighted_avg_speed
    
    def calculate_final_ema_speed(self, speeds, alpha=0.3):
        """
        Calculate the final Exponential Moving Average (EMA) speed for a sequence of speeds,
        emphasizing recent speeds more heavily.

        Args:
            speeds (torch.Tensor): A tensor of shape (sequence_length,) containing the speed
                                values for which to calculate the final EMA speed.
            alpha (float): The smoothing factor used in the EMA calculation, within the range (0, 1).
                        A higher alpha discounts older observations faster.

        Returns:
            float: The final EMA speed value for the sequence.
        """
        if len(speeds) == 0:
            return None  # Handle empty speed sequences

        ema_speed = speeds[0]  # Initialize the EMA with the first speed value

        for i in range(1, len(speeds)):
            ema_speed = alpha * speeds[i] + (1 - alpha) * ema_speed

        return ema_speed

    def transform_tr(self):
        return transforms.Compose([
            transforms.Resize((self.img_width, self.img_height)),
            # transforms.RandomHorizontalFlip(),
            # tr.RandomGaussianBlur(),
            # tr.RandomGaussianNoise(),
            # tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self.dynamic_normalization(x)),
        ])

    def transform_val(self):
        return transforms.Compose([
            transforms.Resize((self.img_width, self.img_height)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self.dynamic_normalization(x)),
        ])

    def transform_ts(self):
        # Assuming transform_ts is meant for testing or as a fallback
        return transforms.Compose([
            transforms.Resize((self.img_width, self.img_height)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self.dynamic_normalization(x)),
        ])

    def dynamic_normalization(self, tensor):
        # Apply dynamic normalization based on the number of channels (C) in the tensor
        if tensor.shape[0] == 1:  # Grayscale image
            return transforms.Normalize(mean=[0.5], std=[0.5])(tensor)
        elif tensor.shape[0] == 3:  # RGB image
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        else:
            raise ValueError("Unsupported channel size: {}".format(tensor.shape[0]))
        
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms  

from PIL import Image
from nuimages import NuImages

from dataset import custom_transforms as tr

class NuImagesDataset(Dataset):
    def __init__(self, dataroot, version='v1.0-mini', img_size=(64, 64), lazy=False, subset=True, train_mode='gray'):
        """
        Initializes the dataset.
        :param dataroot: Root directory where 'nuimages-v1.0-all-metadata' and 'nuimages-v1.0-all-samples' are located.
        :param version: Version of the dataset to use, corresponds to metadata version.
        """
        
        self.split = version
        self.subset = subset
        self.train_mode = train_mode

        # Correctly set the metadata path
        self.metadata_path = os.path.join(dataroot, 'nuimages-v1.0-all-metadata')
        self.samples_root = os.path.join(dataroot, 'nuimages-v1.0-all-samples')
        self.sweeps_root = os.path.join(dataroot, 'nuimages-v1.0-all-sweeps-cam-front')
        

        self.nuim_samples = NuImages(dataroot=self.metadata_path, version=version, verbose=False, lazy=lazy)  # samples annotated
        self.nuim_sweeps = NuImages(dataroot=self.metadata_path, version=version, verbose=False, lazy=lazy)  # sweeps samples raw
        
        # Find the sensor_token for CAM_FRONT
        cam_front_sensor_token = [sensor['token'] for sensor in self.nuim_samples.sensor if sensor['channel'] == 'CAM_FRONT'][0]
        
        # Find calibrated_sensor_tokens associated with CAM_FRONT 
        cam_front_calibrated_tokens = [cs['token'] for cs in self.nuim_samples.calibrated_sensor if cs['sensor_token'] == cam_front_sensor_token]
        
        # Filter and sort sample_data tokens by timestamp and include only those with CAM_FRONT calibrated_sensor_token and is_key_frame
        self.sample_data_tokens = sorted(
            [sd['token'] for sd in self.nuim_samples.sample_data if  sd['calibrated_sensor_token'] in cam_front_calibrated_tokens and sd['is_key_frame']],
            key=lambda token: self.nuim_samples.get('sample_data', token)['timestamp']
        )
        
        # Take only subset
        if self.subset: 
            self.sample_data_tokens = self.sample_data_tokens[:12]

        self.img_mean= [0.485, 0.456, 0.406]
        self.img_std= [0.229, 0.224, 0.225]
        # Resize
        self.img_height=img_size[0] # (h,w)
        self.img_width=img_size[1] # (h,w)
        

        # Choose the appropriate transformation sequence
        self.selected_transform = {
            'v1.0-train': self.transform_tr(),
            'v1.0-val': self.transform_val(),
            'v1.0-test': self.transform_ts(),
            'v1.0-mini': self.transform_tr(),
        }.get(version, self.transform_ts())  # Use the selected transformation

        print('FRONT_CAM only considered', len(self.sample_data_tokens))

    def __len__(self):
        return len(self.sample_data_tokens) - 1

    def __getitem__(self, idx):
        current_token = self.sample_data_tokens[idx]
        current_sample_data = self.nuim_samples.get('sample_data', current_token)
        
        # Load and transform current image
        current_image = self.load_and_transform_image(current_sample_data)
        current_speed = self.get_speed(current_token)
        
        # Initialize list to hold all frames
        frames = []
        
        # Helper function to load and append frames with padding if necessary
        def safe_load_and_pad(token, direction='next', pad_frames=None):
            if token and pad_frames is not None:  # If token is valid and we have frames to pad with
                sample_data = self.nuim_sweeps.get('sample_data',token)
                if sample_data:
                    frame = self.load_and_transform_image(sample_data)
                    pad_frames.append(frame)
                    return sample_data[direction]  # Return the next or prev token
                else:
                    return None
            return None  # Return None if token is None or empty, or sample_data does not exist

        # Retrieve and pad the 6 previous frames
        prev_token = current_sample_data.get('prev', None)
        prev_frames = []
        for _ in range(6):
            prev_token = safe_load_and_pad(prev_token, 'prev', prev_frames)
        
        # Ensure we have 6 frames, padding if necessary
        while len(prev_frames) < 6:
            prev_frames.insert(0, prev_frames[0] if prev_frames else current_image)
        
        # Retrieve and pad the 6 next frames
        next_token = current_sample_data.get('next', None)
        next_frames = []
        for _ in range(6):
            next_token = safe_load_and_pad(next_token, 'next', next_frames)
        
        # Ensure we have 6 frames, padding if necessary
        while len(next_frames) < 6:
            next_frames.append(next_frames[-1] if next_frames else current_image)
        
        # Combine all frames
        frames = prev_frames[::-1] + [current_image] + next_frames
        
        # Stack all images into a single tensor using torch.stack
        stacked_frames = torch.stack(frames).float()
        
        return stacked_frames, current_speed


    def load_and_transform_image(self, sample_data):
        
        # Adjust the image path to point to the samples root.
        image_path = os.path.join(self.samples_root, sample_data['filename'])
        

        if self.train_mode =='gray':
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')


        if image is None:
            raise FileNotFoundError(f"Unable to load image at {image_path}. Check file path and integrity.")
                
        # Apply the selected transformation sequence
        image = self.selected_transform(image)
        
        return image

    def get_speed(self, sample_data_token):
        sample_data = self.nuim_samples.get('sample_data', sample_data_token)
        ego_pose = self.nuim_samples.get('ego_pose', sample_data['ego_pose_token'])
        return ego_pose['speed']


    def transform_tr(self):
        return transforms.Compose([
            transforms.Resize((self.img_width, self.img_height)),
            transforms.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.RandomGaussianNoise(),
            tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
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
        # Assuming this is for testing; similar to validation but can be adjusted as needed
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
        

    # def transform_tr(self):
    #     custom_transforms = transforms.Compose([
    #         transforms.Resize(size=(self.img_width, self.img_height)),
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),  # Convert image to tensor before normalization
    #         transforms.Normalize(mean=[0], std=[1]),  # Example values for grayscale normalization
    #     ])
    #     return custom_transforms

    # def transform_val(self):
    #     return transforms.Compose([
    #         transforms.Resize(size=(self.img_width, self.img_height)),
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.ToTensor(),  # Convert image to tensor before normalization
    #         transforms.Normalize(mean=[0], std=[1]),  # Example values for grayscale normalization
    #     ])

    # def transform_ts(self):
    #     return transforms.Compose([
    #         transforms.Resize(size=(self.img_width, self.img_height)),
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.ToTensor(),  # Convert image to tensor before normalization
    #         transforms.Normalize(mean=[0], std=[1]),  # Example values for grayscale normalization
    #     ])

    
    # def transform_tr(self):
    #     return transforms.Compose([
            
    #             tr.RandomHorizontalFlip(),
    #             tr.RandomGaussianBlur(),
    #             tr.RandomGaussianNoise(),
    #             tr.Resize(size=(self.img_width, self.img_hight)),
    #             tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #             tr.Normalize(mean=self.img_mean, std=self.img_std),
    #             tr.ToTensor()]) 
    
    # def transform_val(self):
    #     return transforms.Compose([
    #             tr.Resize(size=(self.img_width, self.img_hight)),
    #             tr.Normalize(mean=self.img_mean, std=self.img_std),
    #             tr.ToTensor()])

    # def transform_ts(self):
    #     return transforms.Compose([
    #             tr.Resize(size=(self.img_width, self.img_hight)),
    #             tr.Normalize(mean=self.img_mean, std=self.img_std),
    #             tr.ToTensor()])

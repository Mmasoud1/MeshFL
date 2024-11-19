import sqlite3
import torch
import zlib
import numpy as np
# from torchvision import transforms

class Scanloader(torch.utils.data.Dataset):
    def __init__(self, db_file, label_type='label', num_cubes=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.label_type = label_type
        self.query = f"SELECT Image, {self.label_type} FROM mindboggle101"
        self.cursor.execute(self.query)
        self.data = self.cursor.fetchall()
        self.len = len(self.data)
        self.num_cubes = num_cubes

        # # Augmentation: Random rotations and flips
        # self.augmentation = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomRotation(degrees=30)  # You can adjust the degree range
        # ])        

    def __len__(self):
        return self.len

    def divide_into_sub_cubes(self, tensor):
        sub_cubes = []
        sub_cube_size = tensor.shape[0] // self.num_cubes  # Assuming the tensor is a cube

        for i in range(self.num_cubes):
            for j in range(self.num_cubes):
                for k in range(self.num_cubes):
                    sub_cube = tensor[
                        i * sub_cube_size: (i + 1) * sub_cube_size,
                        j * sub_cube_size: (j + 1) * sub_cube_size,
                        k * sub_cube_size: (k + 1) * sub_cube_size
                    ].clone()
                    sub_cubes.append(sub_cube)

        sub_cubes = torch.stack(sub_cubes, 0)
        return sub_cubes

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = zlib.decompress(sample[0])
        image_tensor = torch.from_numpy(np.copy(np.frombuffer(image, dtype=np.float32)).reshape((256, 256, 256)))

        # # Apply data augmentation to image and label
        # image_tensor = self.augmentation(image_tensor)        

        # Normalize image tensor
        # image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

        label = zlib.decompress(sample[1])
        label_tensor = torch.from_numpy(np.copy(np.frombuffer(label, dtype=np.float32)).reshape((256, 256, 256)))




        # return self.divide_into_sub_cubes(image_tensor.to(self.device)), self.divide_into_sub_cubes(label_tensor.to(self.device))

        # if the image_tensor has shape [C, D, H, W] (where C is the number of channels, D is depth, H is height, 
        # and W is width for 3D data), applying unsqueeze(0) will change its shape to [1, C, D, H, W], 
        # where 1 is the batch size. 
              
        return image_tensor.to(self.device), label_tensor.to(self.device)

      
    def split_dataset(self):
        train_size = int(0.75 * self.len)
        valid_size = int(0.15 * self.len)
        train_data, valid_data, infer_data = torch.utils.data.random_split(self, [train_size, valid_size, self.len - train_size - valid_size])
        return train_data, valid_data, infer_data

    def get_loaders(self, batch_size=1, shuffle=True, num_workers=0):
        train_data, valid_data, infer_data = self.split_dataset()
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        infer_loader = torch.utils.data.DataLoader(infer_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, valid_loader, infer_loader
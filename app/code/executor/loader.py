import sqlite3
import torch
import zlib
import json
import os
import numpy as np

class Scanloader(torch.utils.data.Dataset):
    def __init__(self, db_file, label_type='label', num_cubes=1, use_split_file=False, split_file="splits.json", subset="train", logger=None):
        """
        A dataset class for loading and handling 3D MRI data and labels.

        Args:
            db_file (str): Path to the SQLite database file.
            label_type (str): Column name for the label data in the database.
            num_cubes (int): Number of subdivisions for the 3D volume (default is 1, meaning no subdivision).
            use_split_file (bool): Whether to use a split file for dataset partitioning.
            split_file (str): Path to the JSON file defining data splits.
            subset (str): Subset to use ('train', 'valid', or 'test').
            logger (GenericLogger): Logger to use for logging messages.
        """
        self.db_file = db_file  # Store the database file path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conn = sqlite3.connect(db_file)  # Use the db_file directly
        self.cursor = self.conn.cursor()
        self.label_type = label_type
        self.num_cubes = num_cubes
        self.use_split_file = use_split_file
        self.split_file = split_file
        self.subset = subset
        self.logger = logger

        # Query the database
        self.query = f"SELECT Image, {self.label_type} FROM mindboggle101"
        self.cursor.execute(self.query)
        self.data = self.cursor.fetchall()
        self.len = len(self.data)

        # Handle data splitting
        if self.use_split_file:
            if not os.path.exists(self.split_file):
                self.create_split_file()
            self.load_split_file()
        else:
            self.indices = list(range(self.len))

    def create_split_file(self):
        """
        Create the splits.json file if it doesn't exist.
        """
        num_train = int(0.75 * self.len)
        num_valid = int(0.15 * self.len)
        num_test = self.len - num_train - num_valid

        splits = {
            "train": list(range(0, num_train)),
            "valid": list(range(num_train, num_train + num_valid)),
            "test": list(range(num_train + num_valid, self.len))
        }

        with open(self.split_file, "w") as f:
            json.dump(splits, f)

        if self.logger:
            self.logger.log_message(f"Created split file: {self.split_file}")

    def load_split_file(self):
        """
        Load the splits.json file and set indices for the current subset.
        """
        with open(self.split_file, "r") as f:
            splits = json.load(f)

        if self.subset not in splits:
            raise ValueError(f"Subset {self.subset} not found in {self.split_file}")
        self.indices = splits[self.subset]

        if self.logger:
            self.logger.log_message(f"Loaded split file. Subset '{self.subset}' has {len(self.indices)} samples.")


    def __len__(self):
        """
        Return the number of samples in the current subset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the input image tensor and the label tensor.
        """
        actual_idx = self.indices[idx]
        sample = self.data[actual_idx]

        # Decompress and reshape the image
        image = zlib.decompress(sample[0])
        image_tensor = torch.from_numpy(np.copy(np.frombuffer(image, dtype=np.float32)).reshape((256, 256, 256)))

        # Decompress and reshape the label
        label = zlib.decompress(sample[1])
        label_tensor = torch.from_numpy(np.copy(np.frombuffer(label, dtype=np.float32)).reshape((256, 256, 256)))

        return image_tensor.to(self.device), label_tensor.to(self.device)

    def get_loaders(self, batch_size=1, shuffle=True, num_workers=0):
        """
        Create DataLoaders for the dataset.

        Args:
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the training dataset.
            num_workers (int): Number of workers for data loading.

        Returns:
            tuple: Training, validation, and test DataLoaders.
        """
        train_data = Scanloader(
            db_file=self.db_file,  # Pass the stored database file path
            label_type=self.label_type,
            num_cubes=self.num_cubes,
            use_split_file=self.use_split_file,
            split_file=self.split_file,
            subset="train",
            logger=self.logger,
        )
        valid_data = Scanloader(
            db_file=self.db_file,  # Pass the stored database file path
            label_type=self.label_type,
            num_cubes=self.num_cubes,
            use_split_file=self.use_split_file,
            split_file=self.split_file,
            subset="valid",
            logger=self.logger,
        )
        test_data = Scanloader(
            db_file=self.db_file,  # Pass the stored database file path
            label_type=self.label_type,
            num_cubes=self.num_cubes,
            use_split_file=self.use_split_file,
            split_file=self.split_file,
            subset="test",
            logger=self.logger,
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, valid_loader, test_loader


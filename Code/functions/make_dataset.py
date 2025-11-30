import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class WeatherDataset(Dataset):
    """
    Weather Type Classification dataset.
    Owner of dataset: NIKHIL NARAYAN, KAGGLE USERNAME: nikhil7280
    Link to dataset: https://www.kaggle.com/datasets/nikhil7280/weather-type-classification
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Load data from CSV file

        # Identify categorical columns
        cat_cols = self.data.select_dtypes(include=["object"]).columns

        # Create label encoders for each categorical column
        self.encoders = {col: LabelEncoder() for col in cat_cols}

        # Apply encoding
        # String to numerical conversion
        for col in cat_cols:
            self.data[col] = self.encoders[col].fit_transform(self.data[col]).astype(int)

        # Store features and labels  
        self.X = self.data.drop(columns=["Weather Type"]).values.astype("float32")
        self.y = self.data["Weather Type"].values.astype("int64")

        # Scale the data
        transformer = MinMaxScaler()
        self.X = transformer.fit_transform(self.X)
        self.transformer = transformer

        self.transform = transform
        self.cat_cols = cat_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.transform:
            X = self.transform(X)

        return X, y
    
    # Method for decoding categorical values
    def decode_value(self, col, encoded_value):
        """Decode a single encoded categorical value."""
        encoder = self.encoders[col]
        return encoder.inverse_transform([encoded_value])[0]
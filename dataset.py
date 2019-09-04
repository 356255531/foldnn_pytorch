from torch.utils import data
import pickle


class TrainDataset(data.Dataset):

    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            [x, y] = pickle.load(f)
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y
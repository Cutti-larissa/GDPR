import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Config:
    def __init__(self, project, date, rede):
        path = "Coco/" # path relativo para os dados utilizados no experimento
        self.name = "Coco_" + rede + "_" + project + "_" + date
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_data_path = path + project + "/train"
        self.val_data_path = path + project + "/val"
        self.epochs = 100
        self.batch_size = 32
        self.patience = 5
        self.test_data_path = path + "Uncensored/test"

        # Transformações
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Dados
        self.train_data = datasets.ImageFolder(self.train_data_path, transform=transform)
        self.val_data = datasets.ImageFolder(self.val_data_path, transform=transform)
        self.test_data = datasets.ImageFolder(self.test_data_path, transform=transform)

        # Loader
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Tamanho
        self.train_size = len(self.train_data)
        self.val_size = len(self.val_data)
        self.test_size = len(self.test_data)

        # Classes
        self.class_names = self.train_data.classes


import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class MIQRDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        """
        csv_file: Caminho para o csv do split (ex: 'data/MIQR-CC-Dataset/splits/train.csv')
        data_dir: Caminho para a pasta 'data/MIQR-CC-Dataset' onde estão as imagens
        """
        self.metadata = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        
        # Mapeamento estrito das 4 classes exigidas no enunciado
        self.class_map = {
            'Biliary_Leaks': 0, 
            'Lithiasis': 1, 
            'Stricture': 2, 
            'Normal': 3
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # O CSV diz "processed/1_image1.png". Juntando ao data_dir, 
        # fica: data/MIQR-CC-Dataset/processed/1_image1.png
        img_path_rel = self.metadata.iloc[idx]['processed_image_path']
        img_path = os.path.join(self.data_dir, img_path_rel)

        # 1. Carregar imagem original
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

        # 2. Aplicar CLAHE (Melhoria de contraste para Raios-X)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)

        # 3. Converter para RGB (Exigência dos modelos SOTA)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)

        # 4. Transformações PyTorch
        if self.transform:
            image = self.transform(image)

        # 5. Extrair e converter a Label (Vou usar uma coluna "Label_Mapped" que vamos criar no notebook)
        label_str = self.metadata.iloc[idx]['Label_Mapped']
        label = torch.tensor(self.class_map[label_str], dtype=torch.long)

        return image, label

# ------------------------------------------------------------------
# Data Augmentation e Transformações
# ------------------------------------------------------------------

# Transformações para TREINO (Com Data Augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),            # Rotações ligeiras até 15 graus
    transforms.RandomHorizontalFlip(p=0.5),   # Espelho horizontal (50% de probabilidade)
    transforms.ColorJitter(brightness=0.2),   # Ajuste de brilho para simular variação no raio-x
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformações para VALIDAÇÃO e TESTE (SEM Augmentation, apenas normalização)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

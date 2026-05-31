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

    def get_labels(self):
        """Retorna uma lista dos labels numéricos de todas as amostras do dataset (sem carregar as imagens)."""
        return [self.class_map[lbl] for lbl in self.metadata['Label_Mapped']]

# ------------------------------------------------------------------
# Data Augmentation e Transformações
# ------------------------------------------------------------------

# Transformações para TREINO V1 (Sem Augmentation - Padrão / Quase sem nada)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformações para TREINO V2 (Com Data Augmentation Forte - Flips, Rotações e Zoom)
train_transforms_strong = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),             # Rotações ligeiras (até 15 graus)
    #transforms.RandomHorizontalFlip(p=0.5),   # Flip horizontal (50% de probabilidade)
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),                  # Translações ligeiras
        scale=(0.90, 1.10)                     # Zoom ligeiro
    ),
    transforms.ColorJitter(
        brightness=0.25,                       # Ajuste de brilho
        contrast=0.25                          # Ajuste de contraste
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformações para VALIDAÇÃO e TESTE (Apenas normalização e redimensionamento)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ------------------------------------------------------------------
# Weighted Random Sampler (para balancear batches)
# ------------------------------------------------------------------

def get_weighted_sampler(dataset):
    """
    Cria um WeightedRandomSampler para balancear a amostragem das classes durante o treino.
    Garante que cada batch tenha representação equilibrada de todas as classes.
    """
    labels = dataset.get_labels()
    
    # Contar instâncias de cada classe (0 a 3)
    class_counts = torch.zeros(4, dtype=torch.float32)
    for lbl in labels:
        class_counts[lbl] += 1
        
    # Peso da classe é o inverso da frequência
    class_weights = 1.0 / class_counts
    
    # Atribuir peso a cada amostra
    sample_weights = torch.tensor([class_weights[lbl] for lbl in labels], dtype=torch.float32)
    
    # Criar sampler com reposição
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


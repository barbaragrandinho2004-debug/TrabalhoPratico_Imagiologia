"""
Ensemble e Test-Time Augmentation (TTA) para inferência avançada.

Esta classe e funções ajudam a extrair previsões mais robustas dos modelos treinados
sem necessidade de re-treino.
- O Ensemble faz a média das probabilidades de vários modelos.
- O TTA faz previsões de múltiplas versões aumentadas da mesma imagem e faz a média.
"""

import os
import sys
# Adicionar a pasta 'src' ao path caso o script seja executado diretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from training.models import get_model
from evaluation.metrics import MetricsEvaluator

def apply_tta(image_tensor, aug_idx):
    """
    Aplica transformações geométricas básicas para Test-Time Augmentation (TTA).
    Supports 4 states:
    0: Original
    1: Horizontal Flip
    2: Vertical Flip
    3: Horizontal + Vertical Flip
    """
    if aug_idx == 0:
        return image_tensor
    elif aug_idx == 1:
        return TF.hflip(image_tensor)
    elif aug_idx == 2:
        return TF.vflip(image_tensor)
    elif aug_idx == 3:
        return TF.hflip(TF.vflip(image_tensor))
    return image_tensor

def get_predictions_with_tta(model, dataloader, device, n_augments=4):
    """
    Obtém previsões (probabilidades softmax e labels) de um modelo usando Test-Time Augmentation.
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # Acumulador de probabilidades do batch para cada tipo de augmentation
            batch_probs = torch.zeros((batch_size, 4), device=device)
            
            for aug_idx in range(n_augments):
                # Aplicar augmentation a cada imagem no batch
                augmented_inputs = torch.stack([apply_tta(img, aug_idx) for img in inputs])
                
                outputs = model(augmented_inputs)
                probs = torch.softmax(outputs, dim=1)
                batch_probs += probs
                
            # Média das probabilidades
            batch_probs = batch_probs / n_augments
            
            all_probs.append(batch_probs.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_probs), torch.cat(all_labels)

def get_ensemble_predictions(models, dataloader, device, use_tta=False, n_augments=4):
    """
    Faz previsões a partir de um conjunto (ensemble) de múltiplos modelos.
    Calcula a média das probabilidades previstas por cada modelo.
    
    Parâmetros:
    - models (list): Lista de modelos PyTorch em modo eval().
    - dataloader (DataLoader): Loader dos dados de teste/validação.
    - device (torch.device): GPU ou CPU.
    - use_tta (bool): Se True, ativa Test-Time Augmentation para cada modelo.
    - n_augments (int): Número de augmentations TTA a aplicar (máximo 4).
    """
    for model in models:
        model.eval()
        
    all_ensemble_probs = []
    all_labels = []
    
    # Processar lote a lote
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            # Acumular probabilidades de todos os modelos para este lote
            lote_probs_acumuladas = torch.zeros((batch_size, 4), device=device)
            
            for model in models:
                if use_tta:
                    # TTA loop para o modelo atual
                    model_tta_probs = torch.zeros((batch_size, 4), device=device)
                    for aug_idx in range(n_augments):
                        augmented_inputs = torch.stack([apply_tta(img, aug_idx) for img in inputs])
                        outputs = model(augmented_inputs)
                        model_tta_probs += torch.softmax(outputs, dim=1)
                    model_probs = model_tta_probs / n_augments
                else:
                    outputs = model(inputs)
                    model_probs = torch.softmax(outputs, dim=1)
                
                lote_probs_acumuladas += model_probs
                
            # Fazer a média entre os modelos
            lote_ensemble_probs = lote_probs_acumuladas / len(models)
            
            all_ensemble_probs.append(lote_ensemble_probs.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_ensemble_probs), torch.cat(all_labels)

def load_models_from_checkpoints(checkpoint_paths, base_model_names, num_classes=4, device='cpu'):
    """
    Carrega múltiplos modelos a partir dos caminhos dos checkpoints.
    
    Exemplo:
        paths = ['checkpoints/best_resnet50.pth', 'checkpoints/best_densenet121.pth']
        names = ['resnet50', 'densenet121']
        models = load_models_from_checkpoints(paths, names, device=device)
    """
    loaded_models = []
    for path, name in zip(checkpoint_paths, base_model_names):
        print(f"A carregar o modelo {name} de {path}...")
        model = get_model(name, num_classes=num_classes, feature_extracting=False)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # Lidar com diferenças de chaves caso os checkpoints tenham outros formatos
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        loaded_models.append(model)
        
    return loaded_models

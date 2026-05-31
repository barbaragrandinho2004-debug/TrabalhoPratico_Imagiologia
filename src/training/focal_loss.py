"""
Focal Loss para classificação com datasets desbalanceados.

A Focal Loss (Lin et al., 2017) reduz o impacto dos exemplos "fáceis"
(tipicamente da classe dominante) e foca o treino nos exemplos mais difíceis
(como as classes minoritárias Biliary_Leaks e Normal).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        """
        Focal Loss para classificação multi-classe.

        Parâmetros:
        - alpha (Tensor, opcional): Pesos por classe. Ex: [2.4, 0.54, 1.0, 1.32]
        - gamma (float): Fator de focagem. Recomenda-se 2.0.
        - label_smoothing (float): Suavização de rótulos (0.0 a 0.1 recomendado).
        - reduction (str): Tipo de redução: 'mean', 'sum' ou 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        inputs: Logits do modelo (antes do softmax) [BatchSize, NumClasses]
        targets: Rótulos inteiros [BatchSize]
        """
        # 1. Calcular a Cross Entropy pura (sem pesos) com label smoothing por amostra
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        
        # 2. pt é a probabilidade real do modelo para a classe correta
        pt = torch.exp(-ce_loss)
        
        # 3. Aplicar a modulação da Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Aplicar os pesos de classe (alpha) no final, se fornecidos
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

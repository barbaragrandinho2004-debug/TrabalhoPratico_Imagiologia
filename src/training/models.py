import torch
import torch.nn as nn
import torchvision.models as models

def set_parameter_requires_grad(model, feature_extracting):
    """
    Se feature_extracting=True, congelamos os pesos das camadas anteriores (transfer learning).
    Caso contrário, todas as camadas serão otimizadas (fine-tuning).
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(model_name="resnet50", num_classes=4, feature_extracting=True):
    """
    Inicializa e devolve o modelo pedido, ajustando a cabeça de classificação para o nosso num_classes.
    
    Parâmetros:
    - model_name: 'resnet50', 'efficientnet_v2_s', ou 'vit_b_16'
    - num_classes: número de classes a classificar (4 no nosso caso: Fugas, Cálculos, Estenose, Normal)
    - feature_extracting: Se True faz transfer learning puro (congela a base). Se False, faz fine-tuning total.
    """
    if model_name == "resnet50":
        # Carrega o ResNet50 com pesos treinados no ImageNet
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        set_parameter_requires_grad(model, feature_extracting)
        
        # O ResNet50 usa model.fc para a última camada. Vamos alterar o output
        num_ftrs = model.fc.in_features
        # Adicionamos dropout para maior robustez ao ruído antes da classificação final
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    elif model_name == "efficientnet_v2_s":
        # EfficientNet V2 é excelente e extremamente rápido/eficiente
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        set_parameter_requires_grad(model, feature_extracting)
        
        # A última camada está no model.classifier
        # O classificador do efficientnet é um Sequential(Dropout, Linear)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "vit_b_16":
        # Vision Transformer: Ótimo para os Attention Maps
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        set_parameter_requires_grad(model, feature_extracting)
        
        # A cabeça do ViT está em model.heads.head
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "densenet121":
        # DenseNet121: Excelente para imagens médicas de raio-X (usado no paper CheXNet!)
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        set_parameter_requires_grad(model, feature_extracting)
        
        # A última camada está em model.classifier
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    else:
        raise ValueError(f"O modelo '{model_name}' não é suportado nesta função.")
        
    return model

    

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class InterpretabilityTools:
    def __init__(self, model, target_layers, use_cuda=True):
        """
        Inicializa as ferramentas de interpretabilidade (Grad-CAM).
        
        Parâmetros:
        - model: modelo PyTorch carregado.
        - target_layers: lista com as últimas camadas convolucionais do modelo de onde se extraem as ativações.
                         (Ex ResNet50: [model.layer4[-1]])
        - use_cuda: usar GPU se disponível.
        """
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Inicializa o objeto GradCAM
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers) # Removed use_cuda, handle manually if needed or update pytorch-grad-cam version
        
    def generate_gradcam(self, input_tensor, rgb_image, target_category=None, save_path=None):
        """
        Gera e visualiza o Grad-CAM heatmap.
        
        Parâmetros:
        - input_tensor: Tensor imagem normalizado para o modelo [1, 3, H, W].
        - rgb_image: Imagem RGB original normalizada entre [0, 1] (tipo numpy array) para sobrepor o heatmap.
                     (Se for um Raio-X em grayscale, deves convertê-lo primeiro para um array 3-canais).
        - target_category: Classe para a qual queremos ver a explicação (0 a 3). Se None, foca-se na classe com maior probabilidade prevista.
        - save_path: Se indicado, grava a imagem no caminho (string).
        
        Retorna:
        - visualização (numpy array da imagem com o heatmap sobreposto).
        """
        # Se target_category for passado, dizemos ao Grad-CAM para usar essa classe
        targets = None
        if target_category is not None:
            targets = [ClassifierOutputTarget(target_category)]
            
        # Obter o heatmap
        # input_tensor deve ter require_grad=True para o ViT/CNN nalgumas versões, mas o wrapper GradCAM trata disso
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # O resultado vem como batch, como passámos só 1 imagem, extraímos a 1a (índice 0)
        grayscale_cam = grayscale_cam[0, :]
        
        # Sobrepor o heatmap à imagem original usando o utilitário do pytorch_grad_cam
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        
        # Gravar ou mostrar
        if save_path:
            plt.imsave(save_path, visualization)
        
        return visualization

def get_target_layers_for_model(model, model_name):
    """
    Função auxiliar para identificar facilmente a target layer baseada no modelo.
    """
    if model_name == "resnet50":
        return [model.layer4[-1]]
    elif model_name == "efficientnet_v2_s":
        return [model.features[-1]]
    elif model_name == "vit_b_16":
        # Para ViT, o Grad-CAM da biblioteca do jacobgil requer a última camada do Encoder
        # O target é tipicamente o model.encoder.layers[-1].ln_1
        return [model.encoder.layers[-1].ln_1]
    else:
        raise ValueError(f"Target layers desconhecidos para {model_name}")

if __name__ == "__main__":
    # Importar para teste
    import torchvision.models as models
    from pytorch_grad_cam import GradCAM
    
    # Exemplo: Iniciar um ResNet50 para testar a pipeline (sem dados reais)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    target_layers = [model.layer4[-1]]
    
    interpreter = InterpretabilityTools(model, target_layers)
    
    # Gerar dummy data
    input_tensor = torch.rand(1, 3, 224, 224)
    rgb_img = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Gerar o heatmap dummy para testar (para classe alvo 0)
    vis = interpreter.generate_gradcam(input_tensor, rgb_img, target_category=0)
    print("Sucesso! Forma da imagem resultante (Grad-CAM):", vis.shape)

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, LayerCAM, EigenGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class InterpretabilityTools:
    def __init__(self, model, target_layers, use_cuda=True, method="gradcam"):
        """
        Inicializa as ferramentas de interpretabilidade.
        
        Parâmetros:
        - model: modelo PyTorch carregado.
        - target_layers: lista com as camadas do modelo de onde se extraem as ativações.
        - use_cuda: usar GPU se disponível.
        - method: método de visualização ('gradcam', 'layercam', 'eigengradcam', 'eigencam').
        """
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Configura o reshape_transform se o modelo for um Vision Transformer (ViT)
        reshape_transform = None
        model_class_name = self.model.__class__.__name__
        self.is_vit = "VisionTransformer" in model_class_name
        if hasattr(self.model, "module") and not self.is_vit:
            self.is_vit = "VisionTransformer" in self.model.module.__class__.__name__
            
        if self.is_vit:
            def vit_reshape_transform(tensor):
                # tensor shape: [batch, seq_len, hidden_dim]
                seq_len = tensor.size(1)
                hidden_dim = tensor.size(2)
                # seq_len = grid_height * grid_width + 1 (devido ao class token)
                num_patches = seq_len - 1
                grid_size = int(np.sqrt(num_patches))
                
                # Remove o class token (primeiro token, índice 0) e faz reshape para [batch, H, W, C]
                result = tensor[:, 1:, :].reshape(tensor.size(0), grid_size, grid_size, hidden_dim)
                # Permuta para o formato [batch, C, H, W]
                result = result.permute(0, 3, 1, 2)
                return result
            reshape_transform = vit_reshape_transform

        # Seleciona a classe do método CAM correspondente
        method = method.lower()
        if method == "gradcam":
            cam_class = GradCAM
        elif method == "layercam":
            cam_class = LayerCAM
        elif method == "eigengradcam":
            cam_class = EigenGradCAM
        elif method == "eigencam":
            cam_class = EigenCAM
        else:
            raise ValueError(f"Método '{method}' desconhecido. Usa: 'gradcam', 'layercam', 'eigengradcam', 'eigencam'")

        # Inicializa o objeto CAM
        self.cam = cam_class(
            model=self.model,
            target_layers=self.target_layers,
            reshape_transform=reshape_transform
        )
        
    def generate_gradcam(self, input_tensor, rgb_image, target_category=None, save_path=None, eigen_smooth=None):
        """
        Gera e visualiza o Grad-CAM heatmap.
        
        Parâmetros:
        - input_tensor: Tensor imagem normalizado para o modelo [1, 3, H, W].
        - rgb_image: Imagem RGB original normalizada entre [0, 1] (tipo numpy array) para sobrepor o heatmap.
                     (Se for um Raio-X em grayscale, deves convertê-lo primeiro para um array 3-canais).
        - target_category: Classe para a qual queremos ver a explicação (0 a 3). Se None, foca-se na classe com maior probabilidade prevista.
        - save_path: Se indicado, grava a imagem no caminho (string).
        - eigen_smooth: Se indicado, reduz o ruído da imagem usando PCA. Default: True se for ViT, False caso contrário.
        
        Retorna:
        - visualização (numpy array da imagem com o heatmap sobreposto).
        """
        # Se target_category for passado, dizemos ao Grad-CAM para usar essa classe
        targets = None
        if target_category is not None:
            targets = [ClassifierOutputTarget(target_category)]
            
        if eigen_smooth is None:
            eigen_smooth = getattr(self, "is_vit", False)

        # Obter o heatmap
        # input_tensor deve ter require_grad=True para o ViT/CNN nalgumas versões, mas o wrapper GradCAM trata disso
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=eigen_smooth
        )
        
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

    elif model_name == "densenet121":
        return [model.features.denseblock4.denselayer16.conv2]
        
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

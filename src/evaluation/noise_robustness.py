import torch
import torchvision.transforms as T
import numpy as np


# ------------------------------------------------------------------ #
#  Transformações de ruído para testar a robustez do modelo           #
#  (Simula variações reais de contraste e artefactos em fluoroscopia) #
# ------------------------------------------------------------------ #

class GaussianNoise:
    """
    Adiciona ruído Gaussiano a um tensor de imagem.
    Simula ruído eletrónico comum em equipamentos de fluoroscopia de baixa dose.
    """
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return f"GaussianNoise(mean={self.mean}, std={self.std})"


class RandomContrastJitter:
    """
    Varia aleatoriamente o contraste da imagem dentro do intervalo [low, high].
    Simula variações de contraste comuns entre diferentes equipamentos de raio-X.
    """
    def __init__(self, low=0.6, high=1.4):
        self.low = low
        self.high = high

    def __call__(self, tensor):
        factor = torch.empty(1).uniform_(self.low, self.high).item()
        return torch.clamp(tensor * factor, 0.0, 1.0)

    def __repr__(self):
        return f"RandomContrastJitter(low={self.low}, high={self.high})"


class SaltAndPepperNoise:
    """
    Adiciona ruído de sal-e-pimenta (pixels pretos e brancos aleatórios).
    Simula artefactos de pixels mortos ou interferências em imagens de fluoroscopia.
    """
    def __init__(self, amount=0.02):
        self.amount = amount  # Percentagem de pixels afetados

    def __call__(self, tensor):
        noisy = tensor.clone()
        # Pixels brancos (sal)
        num_salt = int(self.amount * tensor.numel() / 2)
        salt_coords = [torch.randint(0, d, (num_salt,)) for d in tensor.shape]
        noisy[salt_coords[0], salt_coords[1], salt_coords[2]] = 1.0
        # Pixels pretos (pimenta)
        pepper_coords = [torch.randint(0, d, (num_salt,)) for d in tensor.shape]
        noisy[pepper_coords[0], pepper_coords[1], pepper_coords[2]] = 0.0
        return noisy

    def __repr__(self):
        return f"SaltAndPepperNoise(amount={self.amount})"


def get_robustness_transforms(noise_type="gaussian"):
    """
    Devolve um conjunto de transformações de robustez para usar no val/test loader
    durante os testes de robustez.

    noise_type: 'gaussian', 'contrast', 'salt_pepper', ou 'combined'
    """
    if noise_type == "gaussian":
        return T.Compose([GaussianNoise(mean=0.0, std=0.05)])

    elif noise_type == "contrast":
        return T.Compose([RandomContrastJitter(low=0.6, high=1.4)])

    elif noise_type == "salt_pepper":
        return T.Compose([SaltAndPepperNoise(amount=0.02)])

    elif noise_type == "combined":
        # Combinação de todos os ruídos — teste de stress máximo
        return T.Compose([
            GaussianNoise(mean=0.0, std=0.04),
            RandomContrastJitter(low=0.7, high=1.3),
            SaltAndPepperNoise(amount=0.01)
        ])
    else:
        raise ValueError(f"noise_type '{noise_type}' não reconhecido. Usa: 'gaussian', 'contrast', 'salt_pepper', 'combined'")


def evaluate_robustness(model, test_loader, evaluator, device, noise_types=None):
    """
    Avalia o modelo sob diferentes tipos de ruído e compara com a performance limpa.
    Retorna um dicionário com os resultados de cada tipo de ruído.

    Parâmetros:
    - model: modelo PyTorch treinado
    - test_loader: DataLoader de teste (sem ruído)
    - evaluator: instância de MetricsEvaluator
    - device: 'cuda' ou 'cpu'
    - noise_types: lista de tipos de ruído a testar. Default: todos os tipos.
    """
    if noise_types is None:
        noise_types = ["clean", "gaussian", "contrast", "salt_pepper", "combined"]

    results = {}
    model.eval()

    for noise_type in noise_types:
        all_preds = []
        all_labels = []

        # Obter o transform de ruído (se não for "clean")
        noise_transform = None if noise_type == "clean" else get_robustness_transforms(noise_type)

        with torch.no_grad():
            for inputs, labels in test_loader:
                # Aplicar ruído aos inputs antes de mandar para o modelo
                if noise_transform is not None:
                    inputs = torch.stack([noise_transform(img) for img in inputs])

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

                all_preds.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metrics = evaluator.evaluate(all_labels, all_preds)
        results[noise_type] = {
            'f1_macro': metrics['f1_macro'],
            'auc_roc': metrics['auc_roc']
        }
    
    clean_f1 = results.get("clean", {}).get("f1_macro", None)
    
    print(f"{'Tipo de Ruído':<20} {'F1-Macro':>10} {'Degradação':>12} {'AUC-ROC':>12}")
    print("-"*60)
    for noise_type, vals in results.items():
        f1 = vals['f1_macro']
        degradation = ""
        auc = vals['auc_roc']
        if clean_f1 is not None and noise_type != "clean":
            diff = f1 - clean_f1
            degradation = f"{diff:+.4f}"
        print(f"{noise_type:<20} {f1:>10.4f} {degradation:>12} {auc:>12.4f}")
    print("="*55)

    
    return results
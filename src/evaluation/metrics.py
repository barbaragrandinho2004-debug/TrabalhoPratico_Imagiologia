import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
import torch
import torch.nn.functional as F

class MetricsEvaluator:
    def __init__(self, class_names):
        """
        Inicializa o avaliador com os nomes das classes para visualização.
        Ex: ['Biliary_Leaks', 'Lithiasis', 'Stricture', 'Normal']
        """
        self.class_names = class_names
        
    def evaluate(self, y_true, y_pred_probs):
        """
        Calcula as métricas F1-Score (macro) e AUC-ROC (ovr).
        y_true: tensores/arrays com as labels verdadeiras (numéricas).
        y_pred_probs: tensores/arrays com as probabilidades (após softmax) de cada classe.
        """
        # Converter para numpy se vier do PyTorch
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred_probs):
            y_pred_probs = y_pred_probs.cpu().numpy()
            
        # As previsões finais são o argmax das probabilidades
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        
        # O professor quer bater o F1-score macro de 0.738 da baseline
        f1_macro = f1_score(y_true, y_pred_labels, average='macro')
        
        # Calcular o AUC-ROC (One-vs-Rest como padrão para multi-classe)
        try:
            auc_roc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
        except ValueError:
            # Caso não haja instâncias de todas as classes no batch (pode acontecer na validação)
            auc_roc = float('nan')
            
        report = classification_report(y_true, y_pred_labels, target_names=self.class_names, zero_division=0)
            
        return {
            'f1_macro': f1_macro,
            'auc_roc': auc_roc,
            'report': report,
            'y_pred_labels': y_pred_labels,
            'y_true': y_true,
            'labels': y_true,
            'probs': y_pred_probs
        }

    def plot_confusion_matrix(self, y_true, y_pred_labels, title="Confusion Matrix", save_path=None):
        """
        Gera o plot visual da Matriz de Confusão usando Seaborn.
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred_labels):
            y_pred_labels = y_pred_labels.cpu().numpy()
            
        cm = confusion_matrix(y_true, y_pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Previsão do Modelo')
        plt.tight_layout()
        
        
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_roc_curves(self, y_true, y_pred_probs, title="Curvas AUC-ROC por Classe", save_path=None):
        """
        Gera o plot das curvas AUC-ROC para cada classe (One-vs-Rest).
        Essencial para o relatório: o professor pede explicitamente este gráfico.
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred_probs):
            y_pred_probs = y_pred_probs.cpu().numpy()

        n_classes = len(self.class_names)
        # Binarizar os labels para OVR (One-vs-Rest)
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        colors = ['#E63946', '#2A9D8F', '#E9C46A', '#457B9D']
        plt.figure(figsize=(9, 6))

        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{class_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Classificação aleatória')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    # Teste rápido
    classes = ['Biliary_Leaks', 'Lithiasis', 'Stricture', 'Normal']
    evaluator = MetricsEvaluator(classes)
    
    # Dados fictícios
    y_true_mock = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred_probs_mock = np.random.rand(8, 4)
    # Normalizar as probabilidades para somarem 1
    y_pred_probs_mock = y_pred_probs_mock / y_pred_probs_mock.sum(axis=1, keepdims=True)
    
    results = evaluator.evaluate(y_true_mock, y_pred_probs_mock)
    print(f"F1 Macro: {results['f1_macro']:.4f}")
    print(f"AUC ROC: {results['auc_roc']:.4f}")
    print(results['report'])

import os
import sys
# Adicionar a pasta 'src' ao path caso o script seja executado diretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from evaluation.metrics import MetricsEvaluator

# Importar livelossplot de forma segura (fallback se não estiver instalado)
try:
    from livelossplot import PlotLosses
except ImportError:
    PlotLosses = None

class TrainerWandb:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, class_names, project_name="Imagiologia_CPRE_Classificacao"):
        """
        Inicializa o treinador que fará tracking automático para o Weights & Biases (WandB).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.evaluator = MetricsEvaluator(class_names)
        self.project_name = project_name
        
    def train_and_evaluate(self, config, epochs=20, save_dir="checkpoints", early_stopping=None, grad_clip=0.0):
        """
        Função principal de treino. Inicia o wandb e corre o loop por X épocas.
        config: Dicionário com hiperparâmetros (ex: lr, batch_size, model_name).
        early_stopping: Objeto EarlyStopping (opcional).
        grad_clip: Norma máxima para clipping de gradientes (opcional, <= 0.0 para desativar).
        """
        # Inicializar a run do WandB
        run = wandb.init(project=self.project_name, config=config)
        
        # Scheduler base para todos os treinos: Cosine Annealing.
        # T_max define o número total de épocas até atingir a LR mínima.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        best_f1 = -1.0
        model_name = config['model_name']
        model_path = os.path.join(save_dir, f"{model_name}.pth")

        plotlosses = PlotLosses(groups={
            '1. Accuracy': ['accuracy', 'val_accuracy'],
            '2. Loss': ['loss', 'val_loss'],
            '3. F1-Macro': ['f1_macro'],
            '4. AUC-ROC': ['auc_roc']
        }) if PlotLosses is not None else None
        
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            # --- FASE DE TREINO ---
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping opcional
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                    
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
                # Calcular previsões para accuracy de treino
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
            train_loss = train_loss / len(self.train_loader.dataset)
            train_acc = train_correct / train_total
            
            # --- FASE DE VALIDAÇÃO ---
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Probabilidades (aplicar softmax para AUC-ROC e para extrair labels)
                    probs = torch.softmax(outputs, dim=1)
                    
                    all_preds.append(probs.cpu())
                    all_labels.append(labels.cpu())
                    
            val_loss = val_loss / len(self.val_loader.dataset)
            
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            # Calcular a accuracy de validação
            val_acc = (all_preds.argmax(dim=1) == all_labels).float().mean().item()
            
            # Calcular métricas usando o nosso MetricsEvaluator
            metrics_results = self.evaluator.evaluate(all_labels, all_preds)
            
            f1_macro = metrics_results['f1_macro']
            auc_roc = metrics_results['auc_roc']
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | F1 Macro: {f1_macro:.4f} | AUC ROC: {auc_roc:.4f} | LR: {current_lr:.2e}")
            
            # Atualizar gráficos do livelossplot
            if plotlosses is not None:
                plotlosses.update({
                    'loss': train_loss,
                    'val_loss': val_loss,
                    'accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'f1_macro': f1_macro,
                    'auc_roc': auc_roc
                })
                plotlosses.send()

            # Tracking para o WandB (Muito importante para os gráficos do relatório!)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "f1_macro": f1_macro,
                "auc_roc": auc_roc,
                "learning_rate": current_lr
            })
            
            # Gravar a melhor epoch deste modelo
            if f1_macro > best_f1:
                best_f1 = f1_macro
                torch.save({
                    'model_name': model_name,
                    'state_dict': self.model.state_dict(),
                    'best_f1': best_f1,
                }, model_path)
                print(f"Novo melhor checkpoint gravado com F1-Macro de {best_f1:.4f}")
                
            # Early Stopping opcional
            if early_stopping is not None:
                score = val_loss if early_stopping.mode == 'min' else f1_macro
                early_stopping(score)
                if early_stopping.early_stop:
                    print(f"\nParagem antecipada (Early Stopping) ativada na época {epoch+1}!")
                    break
                
            # Atualizar o scheduler base (Cosine Annealing)
            scheduler.step()
                
        # No final, submeter o modelo gravado para o cloud artifacts do wandb (opcional mas bom)
        artifact = wandb.Artifact(f'{model_name}', type='model')
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        
        # Carregar o melhor checkpoint deste modelo
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'])
        print(f"\nTreino de {model_name} terminado. Melhor F1 Macro: {best_f1:.4f}")
        
        # Encerra o tracking do wandb
        wandb.finish()
        
        return best_f1

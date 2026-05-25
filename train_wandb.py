import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os

from metrics import MetricsEvaluator

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
        
    def train_and_evaluate(self, config, epochs=20, save_dir="checkpoints"):
        """
        Função principal de treino. Inicia o wandb e corre o loop por X épocas.
        config: Dicionário com hiperparâmetros (ex: lr, batch_size, model_name).
        """
        # Inicializar a run do WandB
        run = wandb.init(project=self.project_name, config=config)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        best_f1 = 0.0
        
        for epoch in range(epochs):
            print(f"\\nEpoch {epoch+1}/{epochs}")
            # --- FASE DE TREINO ---
            self.model.train()
            train_loss = 0.0
            
            # Usar o tqdm para mostrar uma barra de progresso no terminal / Colab
            loop = tqdm(self.train_loader, leave=True)
            for inputs, labels in loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                loop.set_description(f"Treino")
                loop.set_postfix(loss=loss.item())
                
            train_loss = train_loss / len(self.train_loader.dataset)
            
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
            
            # Calcular métricas usando o nosso MetricsEvaluator
            metrics_results = self.evaluator.evaluate(all_labels, all_preds)
            
            f1_macro = metrics_results['f1_macro']
            auc_roc = metrics_results['auc_roc']
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1 Macro: {f1_macro:.4f} | AUC ROC: {auc_roc:.4f}")
            
            # Tracking para o WandB (Muito importante para os gráficos do relatório!)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "f1_macro": f1_macro,
                "auc_roc": auc_roc
            })
            
            # Gravar o melhor modelo
            if f1_macro > best_f1:
                best_f1 = f1_macro
                model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"Novo melhor modelo gravado com F1-Macro de {best_f1:.4f}")
                
        # No final, submeter o modelo gravado para o cloud artifacts do wandb (opcional mas bom)
        artifact = wandb.Artifact('best_model', type='model')
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        
        # Gerar e gravar matriz de confusão final no Colab usando o melhor modelo
        self.model.load_state_dict(torch.load(model_path))
        print("\\nTreino terminado. Melhor F1 Macro obtido:", best_f1)
        
        # Encerra o tracking do wandb
        wandb.finish()
        
        return best_f1

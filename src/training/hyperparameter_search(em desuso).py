import os
import sys
# Adicionar a pasta 'src' ao path caso o script seja executado diretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from training.models import get_model
from training.train_wandb import TrainerWandb

def run_grid_search(
    model_name="densenet121",
    train_loader=None,
    val_loader=None,
    class_names=None,
    class_weights=None,
    device=None,
    lrs=None,
    optimizers_to_try=None,
    epochs=20,
    save_dir="checkpoints/grid_search",
    final_save_dir="checkpoints"
):
    """
    Executa uma pesquisa em grelha (Grid Search) simples variando a Taxa de Aprendizagem (LR)
    e o Otimizador, com decaimento CosineAnnealingLR integrado.
    
    No final, copia automaticamente o melhor checkpoint para a pasta principal.
    
    Retorna:
        pd.DataFrame com a tabela de resultados.
    """
    if lrs is None:
        lrs = [1e-3, 1e-4]
    if optimizers_to_try is None:
        optimizers_to_try = ["Adam", "AdamW"]
        
    grid_results = []
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(final_save_dir, exist_ok=True)
    
    for lr in lrs:
        for opt_name in optimizers_to_try:
            # Nome único para identificar este run
            unique_model_name = f"{model_name}_GS_{opt_name}_{lr}"
            
            print(f"\n{'='*60}")
            print(f" Treinando Grid Search: {unique_model_name}")
            print(f"{'='*60}")
            
            # 1. Carregar arquitetura fresca
            model = get_model(model_name, num_classes=len(class_names), feature_extracting=False)
            
            # 2. Configurar otimizador
            if opt_name == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif opt_name == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            elif opt_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            else:
                raise ValueError(f"Otimizador '{opt_name}' não é suportado nesta função de Grid Search.")
                
            # 3. Configuração do WandB
            config = {
                'model_name': unique_model_name,
                'learning_rate': lr,
                'optimizer': opt_name,
                'scheduler': 'CosineAnnealingLR',
                'base_model': model_name,
                'batch_size': train_loader.batch_size,
                'epochs':epochs,
                'transfer_learning': False,   # False = fine-tuning completo
                'class_weights': True
            }
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
            # 4. Instanciar e correr treino
            trainer = TrainerWandb(model, train_loader, val_loader, criterion, optimizer, device, class_names)
            best_f1 = trainer.train_and_evaluate(config, epochs=epochs, save_dir=save_dir)
            
            grid_results.append({
                "LR": lr,
                "Optimizer": opt_name,
                "Filename": f"{unique_model_name}.pth",
                "Best_Val_F1_Macro": best_f1
            })

    # Criar DataFrame e ordenar por F1 de validação
    df_results = pd.DataFrame(grid_results)
    df_results = df_results.sort_values(by="Best_Val_F1_Macro", ascending=False)
    
    print("\n" + "="*55)
    print("             TABELA DE RESULTADOS DO GRID SEARCH")
    print("="*55)
    print(df_results.to_string(index=False))
    print("="*55)
    
    # 5. Copiar o checkpoint vencedor para a pasta final
    best_run = df_results.iloc[0]
    source_checkpoint = os.path.join(save_dir, best_run["Filename"])
    destination_checkpoint = os.path.join(final_save_dir, f"best_{model_name}.pth")
    
    shutil.copy(source_checkpoint, destination_checkpoint)
    
    print(f"\n🏆 Vencedor: LR = {best_run['LR']:.0e} com {best_run['Optimizer']}")
    print(f"\n(F1-Macro: {best_run['Best_Val_F1_Macro']:.4f})")
    
    return df_results

#1. Importações Básicas

import os
import sys
# Adicionar a pasta 'src' ao path caso o script seja executado diretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# Importar as ferramentas e transformações que criaste no data_setup.py
from preprocessing.data_setup import MIQRDataset, train_transforms, val_transforms


#2. Escrever as Funções do Motor de Treino (train_step e val_step)

# Aqui definimos a lógica matemática de aprendizagem. 
# A função de treino faz o backpropagation (aprende com os erros), e a função de validação apenas "observa" e avalia, sem alterar os pesos da rede.

def train_step(modelo, dataloader, criterio, otimizador, device):
    """Motor de aprendizagem: faz forward pass, calcula o erro e ajusta os pesos."""
    modelo.train()
    perda_total = 0.0
    corretas = 0
    total = 0
    
    for batch_idx, (imagens, labels) in enumerate(dataloader):
        imagens, labels = imagens.to(device), labels.to(device)
        
        otimizador.zero_grad()              # 1. Limpar gradientes antigos
        previsoes = modelo(imagens)         # 2. Fazer a previsão
        perda = criterio(previsoes, labels) # 3. Calcular o erro (com os pesos da rede)
        perda.backward()                    # 4. Backpropagation (aprender)
        otimizador.step()                   # 5. Atualizar os pesos da rede
        
        perda_total += perda.item()
        
        # Imprimir o progresso a cada 10 batches para não encher o ecrã
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}/{len(dataloader)} | Perda: {perda.item():.4f}")
            
    return perda_total / len(dataloader)

def val_step(modelo, dataloader, criterio, device):
    """Motor de avaliação: testa o modelo em dados não vistos (Validação/Teste)."""
    modelo.eval()
    perda_total = 0.0
    corretas = 0
    total = 0
    
    with torch.no_grad(): # Desliga o cálculo de gradientes (poupa imensa memória)
        for imagens, labels in dataloader:
            imagens, labels = imagens.to(device), labels.to(device)
            previsoes = modelo(imagens)
            perda = criterio(previsoes, labels)
            
            perda_total += perda.item()
            _, prever_classes = torch.max(previsoes, 1)
            total += labels.size(0)
            corretas += (prever_classes == labels).sum().item()
            
    accuracy = corretas / total
    return perda_total / len(dataloader), accuracy

#3. DataLoaders e Arquitetura CNN

#Aqui carregamos os 3 CSVs que dividimos no notebook. Usamos um batch_size de 16 para garantir que o teu computador local não fica sem RAM e carrega a ResNet18.

def main():
    # 1. Configurar Dispositivo (GPU se existir, senão Processador)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar no dispositivo: {device}\n")

    # Caminhos dos CSVs guardados na Tarefa 1
    data_dir = os.path.join("..", "data", "MIQR-CC-Dataset")
    train_csv = os.path.join(data_dir, "splits", "train.csv")
    val_csv = os.path.join(data_dir, "splits", "val.csv")
    test_csv = os.path.join(data_dir, "splits", "test.csv") 

    # 2. Criar os Datasets e os DataLoaders (Treino, Val, Teste)
    print("A preparar os DataLoaders...")
    dataset_treino = MIQRDataset(train_csv, data_dir, transform=train_transforms)
    dataset_val = MIQRDataset(val_csv, data_dir, transform=val_transforms)
    dataset_teste = MIQRDataset(test_csv, data_dir, transform=val_transforms)

    loader_treino = DataLoader(dataset_treino, batch_size=16, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)
    loader_teste = DataLoader(dataset_teste, batch_size=16, shuffle=False)

    # 3. Definir a Arquitetura (ResNet-18)
    print("A inicializar a ResNet-18...")
    modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Ajustar a última camada para prever apenas as nossas 4 classes de Imagiologia
    num_funcionalidades = modelo.fc.in_features
    modelo.fc = nn.Linear(num_funcionalidades, 4)
    modelo = modelo.to(device)

    # 4. Injetar os Pesos
    pesos_classes = torch.tensor([2.4043, 0.5454, 1.0055, 1.3230], dtype=torch.float32).to(device)
    criterio = nn.CrossEntropyLoss(weight=pesos_classes)
    otimizador = optim.Adam(modelo.parameters(), lr=0.001)
    
    #5. Loop de Treino Local

    #Para terminar a função main, montamos o loop. Como isto é só a "Baseline Local" para garantires que o código funciona e que os tensores estão corretos, vamos correr apenas 3 épocas.

    EPOCHS = 3 
    print("\nA iniciar o treino da baseline...")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Época {epoch+1}/{EPOCHS} ---")
        
        # Treinar nos dados de Treino
        perda_treino = train_step(modelo, loader_treino, criterio, otimizador, device)
        
        # Avaliar nos dados de Validação
        perda_val, acc_val = val_step(modelo, loader_val, criterio, device)
        
        print(f"Fim da Época {epoch+1} | Perda Treino: {perda_treino:.4f} | Perda Val: {perda_val:.4f} | Accuracy Val: {acc_val*100:.2f}%")

    print("\nBaseline local concluída com sucesso!")

if __name__ == "__main__":
    main()
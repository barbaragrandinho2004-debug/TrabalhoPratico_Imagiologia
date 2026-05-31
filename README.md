# Classificação Automática de Imagens Fluoroscópicas de CPRE via Deep Learning

**Unidade Curricular:** Imagiologia Médica  
**Universidade do Minho** — Mestrado em Engenharia Informática  
**Ano letivo:** 2025/2026

---

## Descrição

Este projeto desenvolve um sistema de classificação automática de imagens fluoroscópicas de **CPRE (Colangiopancreatografia Retrógrada Endoscópica)** utilizando Deep Learning. O sistema é capaz de classificar imagens em 4 categorias diagnósticas:

| Classe | Descrição |
|---|---|
| `Biliary_Leaks` | Fugas de bílis |
| `Lithiasis` | Cálculos/Pedras biliares |
| `Stricture` | Estenoses dos ductos |
| `Normal` | Achados normais |

O trabalho tem como **baseline** o repositório oficial de Mónica Martins ([MIQR-CC-Dataset](https://github.com/monicaccmartins/MIQR-CC-Dataset)) que obteve um F1-Macro de **0.738** com EfficientNet-B7, e implementa várias otimizações sobre essa baseline.

---

## Dataset

O dataset utilizado é o **MIQR-CC (Multimodal Image Quality and Risk in ERCP and Colonoscopy)**, disponível em:

> 📦 https://doi.org/10.6084/m9.figshare.31079236

Após download, a estrutura de pastas esperada é:

```
data/
└── MIQR-CC-Dataset/
    ├── processed/          ← imagens pré-processadas (.png)
    └── splits/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

Os ficheiros CSV devem conter pelo menos as colunas `processed_image_path` e `Label_Mapped`.

---

## Estrutura do Repositório

```
TrabalhoPratico_Imagiologia/
├── data/                               ← dataset (não incluído no zip; ver secção Dataset)
├── notebooks/
│   ├── 01_eda_dados.ipynb              ← Análise Exploratória dos Dados (EDA)
│   ├── 02_modelos_avancados.ipynb      ← Configuração e fluxo de treino dos modelos
│   ├── 03_avaliacao_melhor_modelo.ipynb ← Avaliação final com os melhores checkpoints de cada modelo
│   └── checkpoints/                    ← Pasta para colocar os pesos descarregados do WandB (.pth)
├── outputs/
│   ├── confusion_matrix_densenet121_normal.png     ← Matriz de confusão DenseNet121 (treinos exploratórios)
│   ├── confusion_matrix_efficientnet_v2_s_normal.png ← Matriz de confusão EfficientNet V2 S (treinos exploratórios)
│   ├── confusion_matrix_vit_b_16_normal.png        ← Matriz de confusão ViT B 16 (treinos exploratórios)
│   ├── confusion_matrix_ensemble.png               ← Matriz de confusão do Ensemble (treinos exploratórios)
│   ├── roc_curves_ensemble.png                     ← Curvas ROC do Ensemble (treinos exploratórios)
|   ├
│   ├── gradcam/                                    ← Mapas de calor Grad-CAM / Layer-CAM (treinos exploratórios)
│   │   ├── gradcam_panel_densenet121.png           
│   │   ├── gradcam_panel_efficientnet_v2_s.png     
│   │   ├── gradcam_panel_vit_b_16.png              
│   │   └── gradcam_*_[classe].png                  
│   └── best_models/                                ← Resultados finais dos melhores checkpoints avaliados
│       ├── confusion_matrix_best_models*.png        ← Matrizes de confusão finais
│       ├── confusion_matrix_ensemble.png           ← Matriz de confusão do Ensemble final
│       ├── roc_curves_ensemble.png          ← Curvas ROC do Ensemble final
│       └── gradcam/                         ← Mapas de calor dos melhores modelos avaliados
│           ├── gradcam_panel_*.png          ← Painéis de ativação
│           └── gradcam_*_[classe].png       ← Imagens individuais
├── src/
│   ├── preprocessing/
│   │   └── data_setup.py               ← MIQRDataset, ROI extraction, transforms, WeightedSampler
│   ├── training/
│   │   ├── models.py                   ← get_model() para EfficientNet V2 S, DenseNet121, ViT B 16
│   │   ├── train_wandb.py              ← Loop de treino com tracking WandB
│   │   ├── focal_loss.py               ← Focal Loss ponderada
│   │   └── early_stopping.py           ← Early Stopping (patience=12, métrica F1-Macro)
│   └── evaluation/
│       ├── metrics.py                  ← F1-Macro, AUC-ROC, matrizes de confusão
│       ├── ensemble.py                 ← Ensemble por média ponderada de modelos
│       ├── gradcam_utils.py            ← Grad-CAM (DenseNet) e Layer-CAM (ViT)
│       └── noise_robustness.py         ← Robustez a ruído gaussiano e sal-e-pimenta
├── requirements.txt
└── README.md
```

---

## Instalação

### Pré-requisitos

- Python 3.9+
- CUDA (opcional, mas recomendado para treino)

### 1. Criar ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

> **Nota:** Para instalar PyTorch com suporte CUDA, consulte [pytorch.org](https://pytorch.org/get-started/locally/) e instale a versão adequada à sua GPU antes de correr o comando acima.

---

## Reprodução dos Resultados

### Passo 1 — Análise Exploratória (EDA)

Abrir e correr o notebook:

```
notebooks/01_eda_dados.ipynb
```

Este notebook inclui:
- Distribuição das classes por split (treino/validação/teste)
- Visualização de exemplos de cada classe
- Análise do desequilíbrio de classes

### Passo 2 — Treino dos Modelos (WandB tracking)

O treino e a otimização dos modelos candidatos são realizados utilizando o pipeline integrado com o WandB. O script de treino e configuração está no notebook:

```
notebooks/02_modelos_avancados.ipynb
```

Nele é configurado o pipeline de data augmentation, as perdas adaptativas (Focal Loss) ou o `WeightedRandomSampler`, e o loop de otimização de pesos via AdamW + decaimento por Cosseno.

---

### Passo 3 — Avaliação Final dos Melhores Modelos

Para avaliar de forma justa e obter os melhores resultados possíveis das experiências registadas no WandB:

1. Aceda ao seu painel no **Weights & Biases (WandB)**.
2. Identifique os melhores checkpoints de cada arquitetura (`densenet121`, `efficientnet_v2_s`, `vit_b_16`) com base no melhor F1-Macro de Validação.
3. Transfira manualmente os respetivos ficheiros `.pth` e guarde-os na pasta local `notebooks/checkpoints/best_models`.
4. Abra e execute o notebook:

```
notebooks/03_avaliacao_melhor_modelo.ipynb
```

Este notebook carrega os melhores pesos descarregados de cada modelo e executa todo o pipeline de avaliação final:
* **Métricas do Teste** — Cálculo de F1-Macro, Precision, Recall e AUC-ROC individual para cada modelo.
* **Ensemble** — Combinação ponderada de previsões para maximizar a generalização.
* **Interpretabilidade Visual** — Geração de mapas de calor (Grad-CAM para DenseNet121 e Layer-CAM para ViT).
* **Robustez ao Ruído** — Avaliação de sensibilidade clínica contra ruído Gaussiano e Sal-e-Pimenta.

---

### (Opcional) Execução via Linha de Comandos (WandB)

O script `src/training/train_wandb.py` permite treinar com tracking automático via [Weights & Biases](https://wandb.ai). Para usar:

```bash
wandb login
python src/training/train_wandb.py
```

---

## Principais Decisões de Design

| Componente | Escolha | Justificação |
|---|---|---|
| **ROI Extraction** | Canny + Hough Transform (abordagem Mónica Martins) | Remove bordas do monitor fluoroscópico sem distorcer a zona clínica |
| **Redimensionamento** | `Resize(224)` + `CenterCrop(224)` | Preserva aspect ratio; não distorce a imagem médica |
| **Augmentation** | Sem flip horizontal/vertical | Lateralidade anatómica é clinicamente relevante em CPRE |
| **Desequilíbrio** | Focal Loss OU WeightedRandomSampler (não ambos) | Combinação causava sobrecompensação e catastrophic forgetting |
| **Otimizador** | AdamW + CosineAnnealingLR | Estável e sem necessidade de grid search exaustivo |
| **Early Stopping** | Patience = 12 épocas (F1-Macro val) | Evita overfitting tardio |
| **Gradient Clipping** | norm = 1.0 | Previne gradientes explosivos no ViT |

---

## Resultados Obtidos

Os resultados detalhados (F1-Macro, AUC-ROC, matrizes de confusão) obtidos pelos melhores modelos individuais e pelo Ensemble final estão documentados no **notebook `03_avaliacao_melhor_modelo.ipynb`** e no relatório técnico entregue.

As visualizações geradas encontram-se em `outputs/`.

---

## Referências

- **Dataset:** Martins, M. et al. *Curated endoscopic retrograde cholangiopancreatography images dataset (MIQR-CC)*. arXiv:2601.16759. https://arxiv.org/abs/2601.16759
- **Baseline:** https://github.com/monicaccmartins/MIQR-CC-Dataset
- **CheXNet:** Rajpurkar, P. et al. *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. 2017.
- **Grad-CAM:** Selvaraju, R.R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017.
# Tema
Classificação Automática de Imagens Fluoroscópicas de CPRE via Deep Learning – Conceção e otimização de uma solução de visão por computador para identificação de patologias biliares e pancreáticas.

## Contexto do Trabalho
A Colangiopancreatografia Retrógrada Endoscópica (CPRE) é um procedimento intervencionista complexo que utiliza fluoroscopia (raios-X em tempo real). A interpretação destas imagens é desafiante e crucial para o sucesso do procedimento. O dataset MIQR-CC (Multimodal Image Quality and Risk in ERCP and Colonoscopy) disponibiliza milhares de imagens curadas e anotadas por especialistas, permitindo o desenvolvimento de sistemas de suporte à decisão clínica (CADx).

O objetivo deste trabalho é desenvolver modelos capazes de classificar automaticamente imagens de CPRE em categorias diagnósticas críticas:
- **Biliary_Leaks** (Bile Leaks, Fugas de bílis)
- **Lithiasis** (Biliary lithiasis, Stones, Cálculos/Pedras)
- **Stricture** (Strictures, Estenoses/Aperto dos ductos)
- **Normal** (Achados normais)

## Objetivos de Aprendizagem
Com a realização deste trabalho prático pretende-se sensibilizar e motivar os alunos para a resolução de problemas concretos utilizando as técnicas abordados na UC aplicadas à imagem médica. Especificamente, pretende-se:
- Dominar o fluxo de trabalho com imagens médicas (DICOM/PNG) e metadados.
- Aplicar, comparar e afinar arquiteturas de DL (e.g. CNNs, Vision Transformers) para classificação multi-classe.
- Lidar com datasets desbalanceados (comum na área médica) através de técnicas de aumento de dados (Augmentation), funções de perda ponderadas e funções de perda assimétricas.
- Implementar métricas de avaliação robustas (F1-Score, Matrizes de Confusão, AUC-ROC).
- Garantir a reprodutibilidade e interpretabilidade dos modelos (ex: Grad-CAM).

## Enunciado
Propõe-se o desenvolvimento de uma solução computacional baseada em Deep Learning para a classificação automática de imagens de CPRE, utilizando como base o artigo "Curated endoscopic retrograde cholangiopancreatography images dataset (MIQR-CC)" (https://arxiv.org/abs/arxiv:2601.16759) e o correspondente dataset disponível em https://doi.org/10.6084/m9.figshare.31079236.

As tarefas incluem:
- **Análise Exploratória:** Estudo da distribuição das classes e preparação do dataset (split treino/validação/teste).
- **Desenvolvimento de uma solução:** Tendo como baseline as soluções e os resultados referidos no artigo e disponíveis em https://github.com/monicaccmartins/MIQR-CC-Dataset.
- **Otimização:** Exploração de técnicas avançadas como Transfer Learning, Fine-tuning de modelos SOTA ou arquiteturas híbridas.
- **Avaliação:** Comparação rigorosa dos resultados face à baseline apresentada no artigo (F1-score macro de 0.738).
- **Interpretabilidade (Obrigatório):** Geração de mapas de calor (heatmaps) para identificar as regiões da imagem que influenciaram a classificação do modelo.

O sistema de IA deve ser capaz de processar imagens de entrada e fornecer:
- **Previsão Multi-classe:** Probabilidades para as 4 classes principais.
- **Robustez ao Ruído:** Lidar com variações de contraste e artefactos comuns em raios-X.
- **Explicação Visual:** Visualização das áreas críticas detetadas (ex: localização da estenose ou do cálculo).

## Metodologia
Os alunos devem basear o seu desenvolvimento no repositório oficial e nos dados disponibilizados:
- **Acesso aos Dados:** O dataset está disponível via Figshare (https://doi.org/10.6084/m9.figshare.31079236). Devem utilizar as versões processadas das imagens.
- **Revisão da Baseline:** Consultar o código disponível no GitHub (https://github.com/monicaccmartins/MIQR-CC-Dataset) para compreender o pipeline de pré-processamento.

### Workflow Sugerido:
1. Pré-processamento (e.g. Redimensionamento, Normalização, CLAHE (Contrast Limited Adaptive Histogram Equalization)).
2. Implementação de estratégias para lidar com o desequilíbrio (ex: Class Weights ou Oversampling).
3. Treino de modelos candidatos.
4. Validação e avaliação utilizando o hold-out disponibilizado.

## Entrega e Avaliação
Os resultados devem ser apresentados num relatório técnico (máx. 10 páginas) escrito seguindo o template da Springer para artigos de conferência, contendo:
- Metodologia de tratamento de imagem (pré-processamento).
- Arquitetura detalhada dos modelos e hiperparâmetros.
- Análise comparativa (Tabelas de métricas e Matrizes de Confusão), devendo salientar o F1-Score (macro) obtido nos casos de teste.
- Exemplos de visualização de interpretabilidade (Grad-CAM).

O relatório, assim como um ficheiro zip com o repositório GitHub usado, com instruções de execução (README) e requisitos (requirements.txt) acompanhado de exemplos e indicações que permitam reproduzir todos os passos realizados assim como os resultados obtidos deverá ser submetido, por um elemento do grupo, na plataforma de e-learning da Universidade do Minho (em "Conteúdo/ Projeto Prático (para avaliação)/ Submissão Projeto Prático"). As submissões deverão ser realizadas até ao final do dia 31 de maio de 2026.

No dia 1 de junho de 2026 decorrerão as sessões de apresentação do trabalho desenvolvido. Os grupos de trabalho deverão submeter antecipadamente as suas apresentações na plataforma de e-learning da Universidade do Minho (em "Conteúdo/ Projeto Prático (para avaliação)/ Submissão apresentação"). Cada grupo disporá de 15 minutos para realizar a apresentação, utilizando os meios que considerar mais adequados.

## Nota Pedagógica
Este trabalho segue a filosofia *challenge-based learning*. Os alunos são encorajados a:
- Tentar superar a baseline do artigo original (F1-score > 0.738).
- Refletir sobre as implicações clínicas de falsos negativos em patologias graves (ex: estenoses).
- Utilizar ferramentas de tracking de experiências (ex: WandB ou MLFlow) para documentar a evolução do treino.

**Observação:** O dataset MIQR-CC é particularmente desafiante devido à semelhança visual entre certas patologias sob fluoroscopia. Sugere-se o uso de técnicas de Attention Maps para verificar se o modelo está focado nos ductos biliares ou em artefactos externos.

## Avaliação por Pares
Cada grupo deve realizar uma análise coletiva da contribuição e do esforço que cada membro dedicou ao desenvolvimento do trabalho. A partir dessa análise, devem ser identificados os membros que trabalharam acima, na média e abaixo da média. Para esta componente de avaliação, é atribuído 1 valor para cada estudante, refletindo a sua contribuição individual para o desenvolvimento desta ferramenta de avaliação.

Assim, um dos membros do grupo deve enviar um e-mail, com os outros membros em CC, para valves@di.uminho.pt. O assunto deve ser "IMG – TG - Avaliação Por Pares".

No corpo do e-mail, relativamente a cada elemento do grupo deve estar indicado o seu delta (o valor a ser adicionado à nota desta componente). Ter em atenção que os deltas podem ser negativos, zero ou positivos e que, dentro de cada grupo, a soma dos deltas deve ser sempre igual a 0.00.

**Exemplo 1 (corresponde a um esforço igual de todos):**
- PG1234 João DELTA = 0
- PG5678 António DELTA = 0
- PG9123 Maria DELTA = 0
- PG4567 Rita DELTA = 0

**Exemplo 2 (António recebe 1 valor adicional, Rita mantém a classificação, João e Maria perdem 0.5 valores cada):**
- PG1234 João DELTA = -0.5
- PG5678 António DELTA = 1
- PG9123 Maria DELTA = -0.5
- PG4567 Rita DELTA = 0

## Código de Conduta
Os participantes do presente trabalho académico declaram ter atuado com integridade e confirmam que não recorreram à prática de plágio nem a qualquer forma de utilização indevida ou falsificação de informações ou resultados em nenhuma das etapas conducente à sua elaboração. Mais declaram que conhecem e respeitaram o Código de Conduta Ética da Universidade do Minho.
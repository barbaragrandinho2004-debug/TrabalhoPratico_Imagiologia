"""
Classe Early Stopping para monitorização e paragem antecipada do treino.

Evita o sobreajuste (overfitting) interrompendo o treino quando a métrica
escolhida (val_loss ou f1_macro) deixa de melhorar durante um número de épocas consecutivas.
"""

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, verbose=False, mode='min'):
        """
        Parâmetros:
        - patience (int): Quantidade de épocas a esperar por melhorias antes de parar.
        - min_delta (float): Variação mínima na métrica para ser considerada uma melhoria.
        - verbose (bool): Se True, imprime mensagens sobre o contador de paragem.
        - mode (str): 'min' se monitorizarmos a loss (melhoria = redução).
                      'max' se monitorizarmos o F1-score/Accuracy (melhoria = aumento).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        """
        Atualiza o contador de early stopping com base no score atual.
        """
        if self.best_score is None:
            self.best_score = current_score
            if self.verbose:
                print(f"EarlyStopping inicializado com score base: {self.best_score:.4f}")
        elif self._is_improvement(current_score):
            if self.verbose:
                print(f"EarlyStopping: Melhoria detetada ({self.best_score:.4f} -> {current_score:.4f}). Reset do contador.")
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping contador: {self.counter} de {self.patience} (Melhor score: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, current_score):
        if self.mode == 'min':
            # Queremos que o score (ex: val_loss) diminua
            return current_score < (self.best_score - self.min_delta)
        elif self.mode == 'max':
            # Queremos que o score (ex: F1-Macro) aumente
            return current_score > (self.best_score + self.min_delta)
        else:
            raise ValueError(f"Modo desconhecido: {self.mode}. Escolha 'min' ou 'max'.")

import logging


class EarlyStopping:
    """Signals to stop the training if validation score doesn't improve after a given patience. 
    Args:
        patience (int): How long to wait after last time the validation score increased
        verbose (bool): If True, prints a message for each validation loss improvement. 
                        Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        mode (str): One of `min`, `max`. In `min` mode, training will stop when the quantity
                    monitored has stopped decreasing; in `max` mode it will stop when the
                    quantity monitored has stopped increasing. Default: `min`
    """
    def __init__(self, patience=7, verbose=True, delta=0, mode='min'):
        
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.improvement = False
        self.delta = delta
        self.mode = mode 

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score

        if self.mode == 'min': 
            score = -score

        elif score < self.best_score + self.delta:
            if self.verbose:
                logging.info(f'EarlyStopping - no improvement: {self.counter}/{self.patience}')
            self.improvement = False
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose: 
                    logging.info(f'EarlyStopping - patience reached: {self.counter}/{self.patience}')
                self.early_stop = True
        else:
            if self.verbose:
                logging.info(f'Early stopping - score improved from {self.best_score:.4f} to {score:.4f}')
            self.improvement = True
            self.best_score = score
            self.counter = 0

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'improvement': self.improvement,
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.improvement = state_dict['improvement']
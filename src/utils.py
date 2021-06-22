class EarlyStopping:
    def __init__(self, patience, verbose):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
    
    def __call__(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss
            
        return False

#todo seed

from abc import ABC

class PruningScheduler(ABC):
    def __init__(self, epochs_before_pruning=None):
        self.epochs_before_pruning = epochs_before_pruning

    @abstractmethod
    def shouldPrune(self):
        pass

class NoPrune(PruningScheduler):
    def shouldPrune(self):
        return False

class PruneAfterEpochs(PruningScheduler):
    epochs_since_pruning = 0
    
    def shouldPrune(self):
        if self.epochs_before_pruning == None:
            raise Error("Epochs before pruning not set")

        if self.epochs_before_pruning <= self.epochs_since_pruning:
            self.epochs_since_pruning = 1
            return True
        
        self.epochs_since_pruning += 1
        return False
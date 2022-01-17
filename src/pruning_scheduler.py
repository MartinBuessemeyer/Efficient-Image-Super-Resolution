from abc import ABC, abstractmethod


class PruningScheduler(ABC):
    def __init__(self, epochs_before_pruning=None):
        self.epochs_before_pruning = epochs_before_pruning

    @abstractmethod
    def should_prune(self):
        pass


class NoPrune(PruningScheduler):
    def should_prune(self):
        return False


class PruneAfterEpochs(PruningScheduler):
    epochs_since_pruning = 0

    def should_prune(self):
        if self.epochs_before_pruning is None:
            raise AttributeError("Epochs before pruning not set")

        if self.epochs_before_pruning <= self.epochs_since_pruning:
            self.epochs_since_pruning = 1
            return True

        self.epochs_since_pruning += 1
        return False

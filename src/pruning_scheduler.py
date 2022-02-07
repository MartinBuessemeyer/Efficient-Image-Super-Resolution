from abc import ABC, abstractmethod


class PruningScheduler(ABC):
    @abstractmethod
    def should_prune(self):
        pass


class NoPrune(PruningScheduler):
    def should_prune(self):
        return False


class PruneAfterEpochs(PruningScheduler):
    def __init__(self, epochs_before_pruning, pruning_interval):
        self.epochs_before_pruning = epochs_before_pruning
        self.pruning_interval = pruning_interval
        self.epochs_since_pruning = 0

    def should_prune(self):
        if self.epochs_before_pruning < 0:
            self.epochs_before_pruning -= 1
            return False

        if self.pruning_interval <= self.epochs_since_pruning:
            self.epochs_since_pruning = 1
            return True

        self.epochs_since_pruning += 1
        return False

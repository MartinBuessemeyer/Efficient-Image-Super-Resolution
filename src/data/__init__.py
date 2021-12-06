from importlib import import_module

from torch.utils.data import ConcatDataset
# from dataloader import MSDataLoader
from torch.utils.data import dataloader


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(args, name=d))

            print(datasets)
            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_validate = self.get_evaluation_loader(args, args.data_validation)
        self.loader_test = self.get_evaluation_loader(args, args.data_test)

    def get_evaluation_loader(self, args, data_modules):
        loader = []
        for module_name in data_modules:
            m = import_module('data.' + module_name.lower())
            testset = getattr(m, module_name)(args, train=False, name=module_name)

            loader.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
        return loader

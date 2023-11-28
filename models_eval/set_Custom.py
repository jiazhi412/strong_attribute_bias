import torch
from dataloader.Custom import CustomDataset


class Model:
    #TODO
    def set_data(self, opt):
        """Set up the dataloaders"""

        train_set = None
        dev_set = None
        test_set = None

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        self.dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=opt["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

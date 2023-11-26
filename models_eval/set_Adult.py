import torch
from dataloader.Adult import AdultDataset


class Model:

    def set_data(self, opt):

        train_set = AdultDataset(path=opt['data_path'], mode=opt['Adult_train_mode'], quick_load=True, bias_name='sex', n_bc=opt['n_bc'], p_bc=opt['p_bc'], balance=opt['balance'])
        self.ce = train_set.get_ce()
        dev_set = AdultDataset(path=opt['data_path'], mode=opt['Adult_test_mode'], quick_load=True, bias_name='sex')
        test_set = AdultDataset(path=opt['data_path'], mode=opt['Adult_test_mode'], quick_load=True, bias_name='sex')

        # small piece for debug
        if opt["debug"]:
            num_train = 200
            num_test = 20
            train_set, dev_set, test_set, _ = torch.utils.data.random_split(train_set, [num_train, num_test, num_test, len(train_set)-num_train-num_test*2])

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=opt['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

'''
Module containing singleton classes for configuration variables
'''

import torch


class HoromaDevice:
    class __Device:
        def __init__(self):
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        def __str__(self):
            return repr(self.device)

    device = None

    def __init__(self):
        if not HoromaDevice.device:
            HoromaDevice.device = HoromaDevice.__Device()

    def __getattr__(self, name):
        return getattr(self.device, name)


DEVICE = HoromaDevice().device

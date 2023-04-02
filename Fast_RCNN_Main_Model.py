import torch
from torch import nn
from collections import OrderedDict

def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Fully_Conection_Model(nn.Module):
    def __init__(self):
        super(Fully_Conection_Model, self).__init__()
        self.First_sibling_layer = nn.Sequential(
            OrderedDict([('Dropout_1', nn.Dropout()),
                         ('Linear_1', nn.Linear(512* 7 * 7, 4096)),
                         ('ReLU_1', nn.ReLU(inplace=True)),

                         ('Dropout_2', nn.Dropout()),
                         ('Linear_2', nn.Linear(4096, 4096)),
                         ('ReLU_2', nn.ReLU(inplace=True)),

                         ('Linear_3',nn.Linear(4096, 21))])#20类别+背景
                        )
        self.Second_Sibling_layer = nn.Sequential(
            OrderedDict([('Dropout_1', nn.Dropout()),
                         ('Linear_1', nn.Linear(512* 7 * 7, 4096)),
                         ('ReLU_1', nn.ReLU(inplace=True)),

                         ('Dropout_2', nn.Dropout()),
                         ('Linear_2', nn.Linear(4096, 4096)),
                         ('ReLU_2', nn.ReLU(inplace=True)),

                         ('Linear_3',nn.Linear(4096, 4))])
                        )
    def forward(self, Model_INPUT):
        Flatten_INPUT = Model_INPUT.view(-1, 512 * 7 * 7)
        Classification_Output = self.First_sibling_layer(Flatten_INPUT)
        Bbox_Regression_Output = self.Second_Sibling_layer(Flatten_INPUT)

        return Classification_Output,Bbox_Regression_Output

device = get_device()
Fast_RCNN_Model = Fully_Conection_Model().to(device)
import torchvision
from torchvision import transforms
Train_data = torchvision.datasets.VOCDetection(root = './VOCDetection',
                                               year = '2012',
                                               image_set = 'train',
                                               download = False,
                                               transform =None,
                                               target_transform = None,
                                               transforms =None)

# VaLidation_data = torchvision. datasets. VOCDetection(root = './VOCDetection',
#                                                 year = '2012',
#                                                 image_set = 'val',
#                                                 download = True,
#                                                 transform = None,
#                                                 target_transform = None,
#                                                 transforms = None)

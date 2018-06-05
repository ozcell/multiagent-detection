import torch as K
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Critic(nn.Module):
    def __init__(self, action_space):
        super(Critic, self).__init__()
        
        input_size = 4096*4 + action_space*2
        hidden_size = 1024
        output_size = 1
        
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        

    def forward(self, s, a, FNet):
        
        glimpse_1, glimpse_2, region, whole = s
        action_1, action_2 = a

        glimpse_1 = FNet(glimpse_1)
        glimpse_2 = FNet(glimpse_2)
        region = FNet(region)
        whole = FNet(whole)
        
        x = K.cat([glimpse_1, glimpse_2, region, whole, action_1, action_2], 1)        
        x = self.FC(x)
        return x
    
    
class Actor(nn.Module):

    def __init__(self, action_space, discrete=True):
        super(Actor, self).__init__()
        
        input_size = 4096*3
        hidden_size = 1024
        output_size = action_space

        self.discrete = discrete
        
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        
    def forward(self, s, FNet):
        
        glimpse, region, whole = s 

        glimpse = FNet(glimpse)
        region = FNet(region)
        whole = FNet(whole)
        
        x = K.cat([glimpse, region, whole], 1)
        if self.discrete:
            x = F.softmax(self.FC(x), dim=1)
        else:
            x = F.tanh(self.FC(x))
        return x


class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()

        self.VGG16 = models.vgg16(pretrained=True).features
        self.FC = models.vgg16(pretrained=True).classifier[0:2]

    def forward(self, x):

        x = self.VGG16(x).view(-1, 25088)
        x = self.FC(x)

        return x

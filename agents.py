import torch as K
import torch.nn as nn

from torchvision import models

class Critic(nn.Module):
    def __init__(self, action_space):
        super(Critic, self).__init__()
        
        input_size = 25088*4 + action_space*2
        hidden_size = 2048
        output_size = 1
        
        self.VGG16 = models.vgg16(pretrained=True).features
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        

    def forward(self, glimpse_1, glimpse_2, region, whole, action_1, action_2):
        
        glimpse_1 = self.VGG16(glimpse_1).view(-1, 25088)
        glimpse_2 = self.VGG16(glimpse_2).view(-1, 25088)
        region = self.VGG16(region).view(-1, 25088)
        whole = self.VGG16(whole).view(-1, 25088)
        
        x = K.cat([glimpse_1, glimpse_2, region, whole, action_1, action_2], 1)        
        x = self.FC(x)
        return x
    
    
class Actor(nn.Module):

    def __init__(self, action_space, discrete=True):
        super(Actor, self).__init__()
        
        input_size = 25088*3
        hidden_size = 2048
        output_size = action_space

        self.discrete = discrete
        
        self.VGG16 = models.vgg16(pretrained=True).features
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        
    def forward(self, glimpse, region, whole):
        
        glimpse = self.VGG16(glimpse).view(-1, 25088)
        region = self.VGG16(region).view(-1, 25088)
        whole = self.VGG16(whole).view(-1, 25088)
        
        x = K.cat([glimpse, region, whole], 1)
        if self.discrete:
            x = self.FC(x)
        else:
            x = self.FC(x)
        return x
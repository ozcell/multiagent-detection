import numpy as np
from PIL import Image, ImageDraw, ImageFont
from bbox_utils import jaccard, relative_to_point

import torch as K
from torchvision.datasets import CocoDetection
import torchvision.transforms as T




fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)

def target_transform(target):
    return np.asarray(target[0].get('bbox'))

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

transform = T.Compose([normalize])

target_transform = lambda target: np.asarray(target[0].get('bbox'))

randint = np.random.randint


class marlod(object):
    def __init__(self, step_size=4, glimpse_size=(128,128), out_size=(224,224)):
        super(marlod, self).__init__()

        self.step_size = step_size 
        self.glimpse_size = glimpse_size    #(x, y)
        self.out_size = out_size            #(x, y)
        
        data = CocoDetection(root = '/home/ok18/Datasets/COCO/train2017',
                             annFile = '/home/ok18/Datasets/COCO/annotations/instances_train2017.json',
                             #transform=transform
                             target_transform=target_transform
                             )
        self.locs = [[0,0], [0,0]]
        self.IoU = 0.
        self.rew = 0
        
        self.batch_iterator = iter(data)
        self.data = data

    def reset(self):
        
        self.img, self.target = next(self.batch_iterator)

        h, w = T.ToTensor()(self.img).shape[1], T.ToTensor()(self.img).shape[2]

        x1, y1 = randint(0, w), randint(0, h)
        x2, y2 = randint(0, w), randint(0, h)
        
        locs = [[x1, y1], [x2, y2]]
        obs = self.get_glimpses(locs)
        rew = self.get_reward(locs)
        
        self.locs = locs
        self.obs = obs
        self.rew = rew
        
        return obs
    
    def get_glimpses(self, locs):
        # option 1: each agent observes its own neighbourhood

        glimpse_1 = T.functional.resized_crop(self.img, locs[0][1]-self.glimpse_size[1]//2, 
                                                        locs[0][0]-self.glimpse_size[0]//2, 
                                                        self.glimpse_size[0], 
                                                        self.glimpse_size[1],
                                                        self.out_size)

        glimpse_2 = T.functional.resized_crop(self.img, locs[1][1]-self.glimpse_size[1]//2, 
                                                        locs[1][0]-self.glimpse_size[0]//2, 
                                                        self.glimpse_size[0], 
                                                        self.glimpse_size[1],
                                                        self.out_size)

        # option 2: both agents observe the covered region

        region = T.functional.resized_crop(self.img, min(locs[0][1],locs[1][1]),
                                                     min(locs[0][0],locs[1][0]),
                                                     abs(locs[0][1]-locs[1][1]),
                                                     abs(locs[0][0]-locs[1][0]),
                                                     self.out_size)
        
        whole = T.Resize(self.out_size)(self.img)

        return T.ToTensor()(glimpse_1), T.ToTensor()(glimpse_2), T.ToTensor()(region), T.ToTensor()(whole)   
    
    def step(self, actions):
        
        locs = self.locs
        
        for i, action in enumerate(actions):
            if np.argmax(action) == 0: locs[i] = locs[i]
            if np.argmax(action) == 1: locs[i][0] = locs[i][0] + self.step_size
            if np.argmax(action) == 2: locs[i][0] = locs[i][0] - self.step_size
            if np.argmax(action) == 3: locs[i][1] = locs[i][1] + self.step_size
            if np.argmax(action) == 4: locs[i][1] = locs[i][1] - self.step_size
        
        obs = self.get_glimpses(locs)
        rew = self.get_reward(locs)
        
        self.locs = locs
        self.obs = obs
        self.rew = rew
        
        return obs, rew
        
    def render(self, with_glimpses=True):

        img = self.img.copy()
        draw = ImageDraw.Draw(img)

        bbox_t = list(relative_to_point(self.target.reshape(1,-1), 'numpy').reshape(4,))
        bbox_p = list((self.locs[0][0], self.locs[0][1], self.locs[1][0], self.locs[1][1]))

        r0 = self.glimpse_size[0]//2
        r1 = self.glimpse_size[1]//2
        r = 5
        
        glimpse_1 = self.locs[0][0]-r0,self.locs[0][1]-r1, self.locs[0][0]+r0, self.locs[0][1]+r1
        glimpse_2 = self.locs[1][0]-r0,self.locs[1][1]-r1, self.locs[1][0]+r0, self.locs[1][1]+r1

        agent_1 = self.locs[0][0]-r,self.locs[0][1]-r, self.locs[0][0]+r, self.locs[0][1]+r
        agent_2 = self.locs[1][0]-r,self.locs[1][1]-r, self.locs[1][0]+r, self.locs[1][1]+r

        draw.rectangle(bbox_t, outline='blue')
        draw.rectangle(bbox_p, outline='red')
        draw.rectangle(glimpse_1, outline='magenta')
        draw.rectangle(glimpse_2, outline='cyan')
        draw.ellipse(agent_1, fill='magenta')
        draw.ellipse(agent_2, fill='cyan')
        
        draw.text((5, 10), "IoU = " + str(self.IoU.item()), font=fnt, fill='black')
        draw.text((5, 30), "Reward = " + str(self.rew), font=fnt, fill='black')
        
        if with_glimpses:
            img_glimpses = Image.new('RGB', (self.out_size[0]*2, self.out_size[1]*2))
            
            img_glimpses.paste(T.ToPILImage()(self.obs[3]), (0, 0))
            img_glimpses.paste(T.ToPILImage()(self.obs[2]), (self.out_size[0], 0))
            img_glimpses.paste(T.ToPILImage()(self.obs[0]), (0, self.out_size[1]))
            img_glimpses.paste(T.ToPILImage()(self.obs[1]), (self.out_size[0], self.out_size[1]))
            
            w, h = img.size
            img_glimpses = T.Resize((h, h))(img_glimpses)
            
            img_all = Image.new('RGB', (w+h, h))
            img_all.paste(img, (0, 0))
            img_all.paste(img_glimpses, (w, 0))
            
            draw = ImageDraw.Draw(img_all)
            draw.text((w,         0), "Whole", font=fnt, fill='yellow')
            draw.text((w+h//2,    0), "Region", font=fnt, fill='red')
            draw.text((w,      h//2), "Agent 1", font=fnt, fill='magenta')
            draw.text((w+h//2, h//2), "Agent 2", font=fnt, fill='cyan')
            
            return img_all
        
        else:
            
            return img
    
    def get_reward(self, locs):
        
        reward = 0
        
        bbox_t = K.Tensor(relative_to_point(self.target.reshape(1,4), 'numpy'))
        bbox_p = K.Tensor(np.sort(np.asarray(locs),axis=0).reshape(1,-1))
        
        IoU = jaccard(bbox_t, bbox_p)
        
        if self.IoU < IoU:
            reward += 1
        else:
            reward -= 1
            
        self.IoU = IoU
        
        return reward
            
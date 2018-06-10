import numpy as np
import skimage.io as io

import torch as K
import torchvision.transforms as T

from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CocoDetection
from bbox_utils import jaccard, relative_to_point


norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

rev_norm = T.Compose([T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
                      T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])])                        

trans = T.Compose([T.ToTensor(), norm])
rev_trans = T.Compose([rev_norm, T.ToPILImage()])
target_trans = lambda target: np.asarray(target[0].get('bbox'))

randint = np.random.randint
fnt = ImageFont.load_default()
#fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)

class marlod(object):
    def __init__(self, step_size=4, discrete=True, glimpse_size=(128,128), out_size=(224,224)):
        super(marlod, self).__init__()

        self.step_size = step_size 
        self.discrete = discrete
        self.glimpse_size = glimpse_size    #(x, y)
        self.out_size = out_size            #(x, y)
        
        data = CocoDetection(root = '/home/ok18/Datasets/COCO/train2017/',
                             annFile = '/home/ok18/Datasets/COCO/annotations/instances_train2017.json',
                             #transform=transform
                             target_transform=target_trans
                             )
        
        self.img, self.target = None, None
        self.locs = [[0,0], [0,0]]
        self.IoU_region = 0.
        self.IoU_agent_1 = 0.
        self.IoU_agent_2 = 0.
        self.rews = np.asarray([0.,0.])

        catIds = data.coco.getCatIds(catNms=['cat'])
        imgIds = data.coco.getImgIds(catIds=catIds )
        data.ids = imgIds
        
        self.batch_iterator = iter(data)
        self.data = data

    def reset(self, new_image=True):
        
        if self.img is None or new_image:
            self.img, self.target = next(self.batch_iterator)

            #Id = self.imgIds[randint(0, len(self.imgIds))]
            #path = self.data.root + self.data.coco.loadImgs(Id)[0]['file_name']
            #self.img = T.ToPILImage()(io.imread(path))
            #self.target = np.asarray(self.data.coco.loadAnns(Id)[0]['bbox'])

        w, h = self.img.size

        
        # initial locations
        #x1, y1 = randint(w/4, 3*w/4), randint(h/4, 3*h/4)
        #x2, y2 = randint(w/4, 3*w/4), randint(h/4, 3*h/4)

        e = randint(4)
        if e == 0:
           x1, y1 = 0+self.glimpse_size[0]/2, 0+self.glimpse_size[1]/2
           x2, y2 = w-self.glimpse_size[0]/2, h-self.glimpse_size[1]/2
        elif e == 1:
           x1, y1 = w-self.glimpse_size[0]/2, 0+self.glimpse_size[1]/2
           x2, y2 = 0+self.glimpse_size[0]/2, h-self.glimpse_size[1]/2
        elif e == 2:
           x1, y1 = 0+self.glimpse_size[0]/2, h-self.glimpse_size[1]/2
           x2, y2 = w-self.glimpse_size[0]/2, 0+self.glimpse_size[1]/2
        elif e == 3:
           x1, y1 = w-self.glimpse_size[0]/2, h-self.glimpse_size[1]/2
           x2, y2 = 0+self.glimpse_size[0]/2, 0+self.glimpse_size[1]/2

        x1 += np.random.randint(2*self.glimpse_size[0])-self.glimpse_size[0]
        y1 += np.random.randint(2*self.glimpse_size[1])-self.glimpse_size[1]
        x2 += np.random.randint(2*self.glimpse_size[0])-self.glimpse_size[0]
        y2 += np.random.randint(2*self.glimpse_size[1])-self.glimpse_size[1]

        locs = [[x1, y1], [x2, y2]]
        obs = self.get_glimpses(locs)
        rews = self.get_rewards(locs)
        
        self.locs = locs
        self.obs = obs
        self.rews = rews
        
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

        return trans(glimpse_1), trans(glimpse_2), trans(region), trans(whole)   
    
    def step(self, actions):
        
        locs = self.locs
        
        #[no action, right, left, down, up]
        for i, action in enumerate(actions):
            if self.discrete:
                if np.argmax(action) == 0: locs[i] = locs[i]

                #to limit the agents inside the images      
                if np.argmax(action) == 1: locs[i][0] = min(locs[i][0] + self.step_size, self.img.size[0]-self.glimpse_size[0])
                if np.argmax(action) == 2: locs[i][0] = max(locs[i][0] - self.step_size, 0+self.glimpse_size[0])
                if np.argmax(action) == 3: locs[i][1] = min(locs[i][1] + self.step_size, self.img.size[1]-self.glimpse_size[1])
                if np.argmax(action) == 4: locs[i][1] = max(locs[i][1] - self.step_size, 0+self.glimpse_size[1])
                
                #free agents, they can go outside the image     
                #if np.argmax(action) == 1: locs[i][0] = locs[i][0] + self.step_size
                #if np.argmax(action) == 2: locs[i][0] = locs[i][0] - self.step_size
                #if np.argmax(action) == 3: locs[i][1] = locs[i][1] + self.step_size
                #if np.argmax(action) == 4: locs[i][1] = locs[i][1] - self.step_size
            else:
                #print('agent_' + str(i) + ': ' + str(action))
                locs[i] = list(np.asarray(locs[i]) + np.asarray(action) * self.step_size)
                locs[i][0] = np.clip(locs[i][0], 0+self.glimpse_size[0], self.img.size[0]-self.glimpse_size[0]) 
                locs[i][1] = np.clip(locs[i][1], 0+self.glimpse_size[1], self.img.size[1]-self.glimpse_size[1]) 
        
        obs = self.get_glimpses(locs)
        rews, done = self.get_rewards(locs)
        
        self.locs = locs
        self.obs = obs
        self.rews = rews
        self.done = done
        
        return obs, rews, done
        
    def render(self, with_glimpses=True):

        img = self.img.copy()
        obs = (self.obs[0].clone(), self.obs[1].clone(), self.obs[2].clone(), self.obs[3].clone())
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
        
        draw.text((5, 10), "IoU_region = " + str(self.IoU_region.item()), font=fnt, fill='black')
        draw.text((5, 30), "Rewards = " + str(self.rews), font=fnt, fill='black')
        
        if with_glimpses:
            img_glimpses = Image.new('RGB', (self.out_size[0]*2, self.out_size[1]*2))
            
            img_glimpses.paste(rev_trans(obs[0]), (0, 0))
            img_glimpses.paste(rev_trans(obs[1]), (self.out_size[0], 0))
            img_glimpses.paste(rev_trans(obs[2]), (0, self.out_size[1]))
            img_glimpses.paste(rev_trans(obs[3]), (self.out_size[0], self.out_size[1]))
            
            w, h = img.size
            img_glimpses = T.Resize((h, h))(img_glimpses)
            
            img_all = Image.new('RGB', (w+h, h))
            img_all.paste(img, (0, 0))
            img_all.paste(img_glimpses, (w, 0))
            
            draw = ImageDraw.Draw(img_all)
            draw.text((w,         0), "Agent 1", font=fnt, fill='magenta')
            draw.text((w+h//2,    0), "Agent 2", font=fnt, fill='cyan')
            draw.text((w,      h//2), "Region", font=fnt, fill='red')
            draw.text((w+h//2, h//2), "Whole", font=fnt, fill='yellow')
            
            return img_all
        
        else:
            
            return img
    
    def get_rewards(self, locs):
        
        
        bbox_t = K.Tensor(relative_to_point(self.target.reshape(1,-1), 'numpy'))
        bbox_p = K.Tensor(np.sort(np.asarray(locs),axis=0).reshape(1,-1))

        
        r0 = self.glimpse_size[0]//2
        r1 = self.glimpse_size[1]//2
        glimpse_1 = K.Tensor([self.locs[0][0]-r0,self.locs[0][1]-r1, self.locs[0][0]+r0, self.locs[0][1]+r1]).view(1,-1)
        glimpse_2 = K.Tensor([self.locs[1][0]-r0,self.locs[1][1]-r1, self.locs[1][0]+r0, self.locs[1][1]+r1]).view(1,-1)
        
        IoU_region = jaccard(bbox_t, bbox_p)
        IoU_agent_1 = jaccard(bbox_t, glimpse_1)
        IoU_agent_2 = jaccard(bbox_t, glimpse_2)

        #print('region: ' + str(IoU_region.item()))
        #print('agent_1: ' + str(IoU_agent_1.item()))
        #print('agent_2: ' + str(IoU_agent_2.item()))

        rewards = np.asarray([0.,0.])
        if self.IoU_region < IoU_region:
            rewards += [1.,1.]
        else:
            rewards -= [1.,1.]

        #if self.IoU_agent_1 < IoU_agent_1:
        #    rewards += [.5,0.]
        #else:
        #    rewards -= [.5,0.]

        #if self.IoU_agent_2 < IoU_agent_2:
        #    rewards += [0.,.5]
        #else:
        #    rewards -= [0.,.5]                
        
        if IoU_region < 0.75:
            done = 0
        else:
            done = 1

        #rewards = [IoU_region+IoU_agent_1, IoU_region+IoU_agent_2]
        
        self.IoU_region = IoU_region
        self.IoU_agent_1 = IoU_agent_1
        self.IoU_agent_2 = IoU_agent_2
        
        return rewards, done

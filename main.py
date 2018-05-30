import numpy as np
import imageio
from marlod import marlod

env = marlod()
obs = env.reset()

render_imgs = []
for i in range(20):
    actions = np.zeros((2,5), dtype=int)
    for i, action in enumerate(actions):
        actions[i,np.random.randint(0,5)] = 1
    
    obs = env.step(actions)
    img = env.render()
    render_imgs.append(np.array(img)) 

imageio.mimsave('render0.gif', render_imgs)  



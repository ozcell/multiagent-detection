import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents import Actor, Critic, Features
from exploration import gumbel_softmax


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class MADDPG(object):
    
    def __init__(self, action_space, optimizer, loss_func, gamma, tau, 
                 discrete=True, regularization=False, dtype=K.float32, device="cuda"):
        
        optimizer, lr = optimizer
        actor_lr, critic_lr = lr
        num_agents = 2
        
        self.loss_func = loss_func
        self.gamma = gamma
        self.tau = tau
        self.discrete = 2
        self.regularization = regularization
        self.dtype = dtype
        self.device = device
        self.num_agents = num_agents
        
        # model initialization
        
        # feature network
        self.FNet = Features().to(device)
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(action_space, discrete).to(device))
            self.actors_target.append(Actor(action_space, discrete).to(device))
            self.actors_optim.append(optimizer(self.actors[i].FC.parameters(), lr = actor_lr))
            
        for i in range(num_agents):
            hard_update(self.actors_target[i], self.actors[i])

        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []

        for i in range(num_agents):
            self.critics.append(Critic(action_space).to(device))
            self.critics_target.append(Critic(action_space).to(device))
            self.critics_optim.append(optimizer(self.critics[i].FC.parameters(), lr = critic_lr))
                
        for i in range(num_agents):
            hard_update(self.critics_target[i], self.critics[i])         
    
    def select_action(self, state, i_agent, exploration=False):
        with K.no_grad():
            mu = self.actors[i_agent](state.to(self.device), self.FNet)
        if self.discrete:
            mu = gumbel_softmax(mu, exploration=exploration)
        else:
            if exploration:
                mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
            
        return mu.clamp(-1, 1) 
                
    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]

        Q = self.critics[i_agent](s, a, self.FNet)

        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](s_[[i,2,3],], self.FNet), exploration=False)

        V[mask] = self.critics_target[i_agent](s_, a_, self.FNet).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r.squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        for i in range(self.num_agents):
            a[i,] = gumbel_softmax(self.actors[i](s[[i,2,3],], self.FNet), exploration=False)

        loss_actor = -self.critics[i_agent](s, a, self.FNet).mean()
        if self.regularization:
            loss_actor += (self.actors[i_agent](s[[i_agent,2,3],])**2, self.FNet).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)
        
        return loss_critic.item(), loss_actor.item()      

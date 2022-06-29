from cmath import nan
import yaml
import isaacgym

import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tasks.cartpole import Cartpole
from isaacgymenvs.tasks import isaacgym_task_map

def orthogonal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight)
        layer.bias.data.fill_(0.0)
    
class ActorCritic(nn.Module):

    def __init__(self, n_obs, n_actions, device):
        super(ActorCritic, self).__init__()
        #Outputs mean value of action
        self.actor = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )
        self.actor.apply(orthogonal_init)
        #We want the variance elements to be trainable, assuming no cross correlation
        self.actor_variance = torch.full((n_actions,), 0.15, device=device)
        
        #Outputs value function
        self.critic = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.critic.apply(orthogonal_init)

    def select_action_logprob(self, state):
            mean = self.get_action_mean(state)
            #covariance = torch.diag(self.actor_diag_covariance)
            prob_dist = torch.distributions.Normal(mean, self.actor_variance)
            action = prob_dist.sample()
            log_prob = prob_dist.log_prob(action)
            return action, log_prob
    
        
def select_action_normaldist(mean, variance, num_envs):
    variance_matrix = variance.repeat((1,num_envs), num_envs).reshape(num_envs,variance.shape[0])
    prob_dist = torch.distributions.Normal(mean, variance_matrix)
    action = prob_dist.sample()
    return action, prob_dist
def select_action_multinormaldist(mean, variance, num_envs):
    #variance_matrix = variance.repeat((1,num_envs), num_envs).reshape(num_envs,variance.shape[0])
    #covariance_matrix = torch.diag(variance_matrix).unsqueeze(dim=0)
    covariance_matrix = torch.diag(variance)
    prob_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    action = prob_dist.sample()
    return action, prob_dist

class PPO():
    def __init__(self):
        #Load task/env specific config
        env = "Cartpole"
        with open("cfg/"+env+".yaml", 'r') as stream:
            try:
                cfg=yaml.safe_load(stream)
                print(cfg)
            except yaml.YAMLError as exc:
                print(exc)
        
        #self.vecenv = Cartpole(cfg, cfg["sim_device"], cfg["graphics_device_id"], 0)
        self.vecenv = isaacgym_task_map[env](cfg, cfg["sim_device"], cfg["graphics_device_id"], 0)
        self.num_obs = self.vecenv.cfg["env"]["numObservations"]
        self.num_actions = self.vecenv.cfg["env"]["numActions"]
        self.num_envs = cfg["env"]["numEnvs"]
        self.device = cfg["rl_device"]

        #Hyperparams

        # Samples collected in total is num_envs*rollout_steps. minibatch_size should be a an integer factor of rollout_steps
        self.rollout_steps = 200
        self.minibatch_size = 10

        self.num_epoch = 5
        self.l_rate = 3e-4 
        self.gamma = 0.99 #Reward discount factor
        self.lambda_ = 0.95 #GAE tuner
        self.epsilon = 0.3 #Clip epsilon

        self.ac = ActorCritic(self.num_obs, self.num_actions, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.l_rate)

        self.tb = SummaryWriter()

        #Allocate our buffers containing rollout_steps amount of data similar to isaacgymenvs vecenv output
        self.obs_buf = torch.zeros((self.rollout_steps, self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.reward_buf = torch.zeros((self.rollout_steps, self.num_envs), device = self.device, dtype=torch.float)
        self.reset_buf = torch.ones((self.rollout_steps+1, self.num_envs), device=self.device, dtype=torch.long)
        #We need rollout_step+1 values for advantage estimation
        self.value_buf = torch.zeros((self.rollout_steps+1, self.num_envs), device=self.device, dtype=torch.float)
        self.action_buf = torch.zeros((self.rollout_steps, self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.log_prob_buf = torch.zeros((self.rollout_steps, self.num_envs), device=self.device, dtype=torch.float)

        #Buffers for last operation
        self.next_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.next_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        

    def run(self):

        #Get first observation
        obs_dict = self.vecenv.reset()
        self.next_obs_buf = obs_dict["obs"]
        
        score = 0
        self.global_step = 0
        while(True):
            
            #Collect rollout
            for step in range(self.rollout_steps):
                self.obs_buf[step] = self.next_obs_buf
                self.reset_buf[step] = self.next_reset_buf
                #Only inference, no grad needed
                with torch.no_grad():
                    self.value_buf[step] = self.ac.critic(self.next_obs_buf).squeeze()
                    action, prob_dist = select_action_multinormaldist(self.ac.actor(self.next_obs_buf), self.ac.actor_variance, self.num_envs)
                self.log_prob_buf[step] = prob_dist.log_prob(action)
                self.next_obs_dict, self.reward_buf[step], self.next_reset_buf, _ = self.vecenv.step(action)
                self.action_buf[step] = action
                self.next_obs_buf = obs_dict["obs"]

                #Calculate score
                score += self.reward_buf[step].mean()

                #Do average reward over 100 timesteps
                if self.global_step % 100 == 0 and self.global_step != 0:
                    self.tb.add_scalar("Score/timestep", score, self.global_step)
                    score = 0
                self.global_step+=1
            
            #Calculate generalized advantage estimate, looping backwards
            with torch.no_grad():
                self.value_buf[self.rollout_steps] = self.ac.critic(self.next_obs_buf).squeeze()
            self.reset_buf[self.rollout_steps] = self.next_reset_buf
            gae_buf = torch.zeros((self.rollout_steps, self.num_envs), device=self.device, dtype=torch.float)
            gae_next = 0
            for step in reversed(range(self.rollout_steps)):
                delta = self.reward_buf[step] + self.gamma * self.value_buf[step+1] * self.reset_buf[step+1] - self.value_buf[step]
                gae_buf[step] =  delta + self.gamma * self.lambda_ * gae_next * self.reset_buf[step+1]
                gae_next = gae_buf[step]
            #Value buf is one element longer than gae_buf, we have bootstrapped next_value
            return_buf = gae_buf + self.value_buf[0:self.rollout_steps]

            #Update network
            shuffled_indicies = np.arange(self.rollout_steps)
            for k in range(self.num_epoch):
                    np.random.shuffle(shuffled_indicies)
                    for mb in range(self.rollout_steps//self.minibatch_size):
                        minibatch_indicies = shuffled_indicies[mb*self.minibatch_size:(mb+1)*self.minibatch_size]
                        actions, prob_dists = select_action_multinormaldist(self.ac.actor(self.obs_buf[minibatch_indicies]), self.ac.actor_variance, self.num_envs)
                        log_probs_new = prob_dists.log_prob(actions)
                        


                        values = self.ac.critic(self.obs_buf[minibatch_indicies]).squeeze()
                        log_prob = self.log_prob_buf[minibatch_indicies]
                        returns = return_buf[minibatch_indicies]
                        advantage = gae_buf[minibatch_indicies]
                        self.update_net(log_prob, log_probs_new, values, returns, advantage)
                    
            

            self.ac.actor_variance *= 0.9

            print("Timestep" + str(self.global_step) + ": Score: " + str(score.data) + ", Action Variance: " + str(self.ac.actor_variance.data.mean()))
            self.tb.add_scalar("Advantage", gae_buf.mean(), self.global_step)

    def update_net(self, log_prob, log_prob_new, values, returns, advantage):
        #Calculate loss function for actor network
        r = torch.exp(log_prob_new - log_prob).squeeze()
        # Loss negated as we want SGD
        l1 = advantage*r
        l2 = torch.clamp(r,1-self.epsilon, 1+self.epsilon)*advantage
        actor_loss = -torch.min(l1, l2).mean()
        
        #Critic loss
        critic_scale = 0.5
        critic_loss = critic_scale*torch.pow(values-returns,2).mean()
        
        critic_loss = F.smooth_l1_loss(values, returns)
        tot_loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        tot_loss.mean().backward()
        #nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
        self.optimizer.step()

        self.tb.add_scalar("Loss/total", tot_loss, self.global_step)
        self.tb.add_scalar("Loss/Actor", actor_loss, self.global_step)
        self.tb.add_scalar("Loss/Critic", critic_loss, self.global_step)

if __name__ == "__main__":
    #os.chdir()
    learner = PPO()
    learner.run()
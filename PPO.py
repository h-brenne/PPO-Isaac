from cmath import nan
import yaml
import isaacgym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from tasks.cartpole import Cartpole

def orthogonal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight)
        layer.bias.data.fill_(0.0)
    
class ActorCritic(nn.Module):

    def __init__(self, n_obs, n_actions):
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
        self.actor_variance = torch.full((n_actions,1), 0.15, device=device)
        
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
    
        
def select_action_normaldist(mean, variance):
    prob_dist = torch.distributions.Normal(mean, variance)
    action = prob_dist.sample()
    return action, prob_dist
def select_action_multinormaldist(mean, variance):
    covariance = torch.diag(variance)
    prob_dist = torch.distributions.MultivariateNormal(mean, covariance)
    action = prob_dist.sample()
    return action, prob_dist

def update_net(log_prob, log_prob_new, values, returns, advantage, ac, tb, global_step):
     #Calculate loss function for actor network
    r = torch.exp(log_prob_new - log_prob).squeeze()
    # Loss negated as we want SGD
    l1 = advantage*r
    l2 = torch.clamp(r,1-epsilon, 1+epsilon)*advantage
    actor_loss = -torch.min(l1, l2).mean()
    
    #Critic loss
    critic_scale = 0.5
    critic_loss = critic_scale*torch.pow(values-returns,2).mean()
    
    tot_loss = actor_loss + critic_loss
    
    optimizer.zero_grad()
    tot_loss.mean().backward()
    nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
    optimizer.step()

    tb.add_scalar("Loss/total", tot_loss, global_step)
    tb.add_scalar("Loss/Actor", actor_loss, global_step)
    tb.add_scalar("Loss/Critic", critic_loss, global_step)

if __name__ == "__main__":
    #Load task/env specific config
    env = "Cartpole"
    with open("cfg/"+env+".yaml", 'r') as stream:
        try:
            cfg=yaml.safe_load(stream)
            print(cfg)
        except yaml.YAMLError as exc:
            print(exc)
    
    vecenv = Cartpole(cfg, cfg["sim_device"], cfg["graphics_device_id"], 0)

    num_obs = vecenv.cfg["env"]["numObservations"]
    num_actions = vecenv.cfg["env"]["numActions"]
    num_envs = cfg["env"]["numEnvs"]
    device = cfg["rl_device"]

    #Hyperparams

    # Samples collected in total is num_envs*rollout_steps. minibatch_size should be a an integer factor of rollout_steps
    rollout_steps = 100
    minibatch_size = 25

    num_epoch = 3
    l_rate = 1e-4 
    gamma = 0.99 #Reward discount factor
    lambda_ = 0.95 #GAE tuner
    epsilon = 0.3 #Clip epsilon

    ac = ActorCritic(num_obs,num_actions).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=l_rate)

    tb = SummaryWriter()

    #Allocate our buffers containing rollout_steps amount of data similar to isaacgymenvs vecenv output
    obs_buf = torch.zeros((rollout_steps, num_envs,num_obs), device=device, dtype=torch.float)
    reward_buf = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float)
    reset_buf = torch.ones((rollout_steps+1, num_envs), device=device, dtype=torch.long)
    #We need rollout_step+1 values for advantage estimation
    value_buf = torch.zeros((rollout_steps+1, num_envs), device=device, dtype=torch.float)
    action_buf = torch.zeros((rollout_steps, num_envs,num_actions), device=device, dtype=torch.float)
    log_prob_buf = torch.zeros((rollout_steps, num_envs,num_actions), device=device, dtype=torch.float)
    #return_buf = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float)
    #Buffers for last operation
    next_obs_buf = torch.zeros((num_envs, num_obs), device=device, dtype=torch.float)
    next_reset_buf = torch.ones(num_envs, device=device, dtype=torch.long)
    
    #Get first observation
    obs_dict = vecenv.reset()
    next_obs_buf = obs_dict["obs"]

    score = 0
    global_step = 0
    while(True):
        
        #Collect rollout
        for step in range(rollout_steps):
            obs_buf[step] = next_obs_buf
            reset_buf[step] = next_reset_buf
            #Only inference, no grad needed
            with torch.no_grad():
                value_buf[step] = ac.critic(next_obs_buf).squeeze()
                action, prob_dist = select_action_normaldist(ac.actor(next_obs_buf), ac.actor_variance)
            log_prob_buf[step] = prob_dist.log_prob(action)
            next_obs_dict, reward_buf[step], next_reset_buf, _ = vecenv.step(action)
            action_buf[step] = action
            next_obs_buf = obs_dict["obs"]

            #Calculate score
            score += reward_buf[step].mean()
            #cumulative_reward = score[reset_buf[step].nonzero()].mean()
            #if(not torch.isnan(cumulative_reward)):
            #    tb.add_scalar("Score/timestep", cumulative_reward, global_step)
            if global_step % 100 == 0 and global_step != 0:
                tb.add_scalar("Score/timestep", score, global_step)
                score = 0
            global_step+=1
        
        #Calculate generalized advantage estimate, looping backwards
        with torch.no_grad():
            value_buf[rollout_steps] = ac.critic(next_obs_buf).squeeze()
        reset_buf[rollout_steps] = next_reset_buf
        gae_buf = torch.zeros((rollout_steps, num_envs), device=device, dtype=torch.float)
        gae_next = 0
        for step in range(rollout_steps-1, -1, -1):
            delta = reward_buf[step] + gamma * value_buf[step+1] * reset_buf[step+1] - value_buf[step]
            gae_buf[step] =  delta + gamma * lambda_ * gae_next *reset_buf[step+1]
            gae_next = gae_buf[step]
        return_buf = gae_buf + value_buf[0:rollout_steps]

        #Update network
        shuffled_indicies = np.arange(rollout_steps)
        for k in range(num_epoch):
                np.random.shuffle(shuffled_indicies)
                for mb in range(rollout_steps//minibatch_size):
                    minibatch_indicies = shuffled_indicies[mb*minibatch_size:(mb+1)*minibatch_size]
                    actions, prob_dists = select_action_normaldist(ac.actor(obs_buf[minibatch_indicies]), ac.actor_variance)
                    log_probs_new = prob_dists.log_prob(actions)
                    


                    values = ac.critic(obs_buf[minibatch_indicies]).squeeze()
                    log_prob = log_prob_buf[minibatch_indicies]
                    returns = return_buf[minibatch_indicies]
                    advantage = gae_buf[minibatch_indicies]
                    update_net(log_prob, log_probs_new, values, returns, advantage, ac, tb, global_step)
                
        

        ac.actor_variance *= 0.9

        print("Timestep" + str(global_step) + ": Score: " + str(score.data) + ", Action Variance: " + str(ac.actor_variance.data.mean()))
        tb.add_scalar("Advantage", gae_buf.mean(), global_step)
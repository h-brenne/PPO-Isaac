#from cmath import nan
import yaml
import isaacgym

from pathlib import Path
import os
from datetime import datetime


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np

#from tasks.cartpole import Cartpole
from isaacgymenvs.tasks import isaacgym_task_map

def scale_rewards(rewards, running_stats)

def orthogonal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight)
        #layer.bias.data.fill_(0.0)
    
class ActorCritic(nn.Module):

    def __init__(self, layer_array, n_obs, n_actions, orthogonal, init_variance, device):
        super(ActorCritic, self).__init__()
        #Outputs mean value of action
        self.actor = nn.Sequential(*self.create_net(layer_array, n_obs, n_actions))
        

        #We want the variance elements to be trainable, assuming no cross correlation
        self.actor_variance = torch.full((n_actions,), init_variance, device=device)
        
        #Outputs value function
        self.critic = nn.Sequential(*self.create_net(layer_array, n_obs, 1))
        if orthogonal:
            self.actor.apply(orthogonal_init)
            self.critic.apply(orthogonal_init)

    def create_net(self, layer_array, input_size, output_size):
        layers = []
        layers.append(nn.Linear(input_size, layer_array[0]))
        layers.append(nn.Tanh())
        for i in range(len(layer_array)-1):
            layers.append(nn.Linear(layer_array[i], layer_array[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_array[-1], output_size))
        return layers
        
def select_action_normaldist(mean, variance):
    prob_dist = torch.distributions.Normal(mean, variance)
    action = prob_dist.sample()
    return action, prob_dist
def select_action_multinormaldist(mean, variance):
    #Torch infers the batch_shape from covariance shape
    covariance_matrix = torch.diag(variance)
    prob_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    action = prob_dist.sample()
    return action, prob_dist

class PPO():
    def __init__(self, env_cfg_file, train_cfg_file, run_name = 'None', checkpoint = 'None'):
        #Load task/env and training/algo specific config
        with open(env_cfg_file, 'r') as stream:
            try:
                env_cfg=yaml.safe_load(stream)
                print(env_cfg)
            except yaml.YAMLError as exc:
                print(exc)
        with open(train_cfg_file, 'r') as stream:
            try:
                train_cfg=yaml.safe_load(stream)
                print(train_cfg)
            except yaml.YAMLError as exc:
                print(exc)
        
        self.now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if run_name == 'None':
            run_name = self.now
        else:
            run_name = run_name #+ "/" + self.now

        self.log_dir = "runs/" + env_cfg["name"] + "/" + run_name
        Path(self.log_dir).mkdir(parents=True, exist_ok=True) #Create dir if it doesn't exist
        
        self.checkpoint = False
        if(checkpoint != 'None'):
            self.checkpoint = True

        #Setup sim
        env_cfg["env"]["numEnvs"] = train_cfg["numEnvs"]

        print("test")
        self.vecenv = isaacgym_task_map[env_cfg["name"]](   env_cfg, 
                                                            env_cfg["rl_device"], 
                                                            env_cfg["sim_device"], 
                                                            env_cfg["graphics_device_id"], 
                                                            env_cfg["headless"], 
                                                            0, 
                                                            env_cfg["force_render"])
        self.num_obs = self.vecenv.cfg["env"]["numObservations"]
        self.num_actions = self.vecenv.cfg["env"]["numActions"]
        self.num_envs = env_cfg["env"]["numEnvs"]
        self.device = env_cfg["rl_device"]

        self.total_updates = train_cfg["total_updates"]

        #Hyperparams
        # Samples collected in total is num_envs*rollout_steps. minibatch_size should be a an integer factor of rollout_steps
        self.rollout_steps = train_cfg["rollout_steps"]
        self.minibatch_size = train_cfg["minibatch_size"]
        print("Minibatch samples: ", self.minibatch_size*self.num_envs)

        self.num_epoch = train_cfg["num_epoch"]
        self.l_rate = train_cfg["learning_rate"]
        self.gamma = train_cfg["gamma"] #Reward discount factor
        self.lambda_ = train_cfg["lambda"] #GAE tuner
        self.epsilon = train_cfg["epsilon_clip"] #Clip epsilon

        self.eval_freq = train_cfg["eval_freq"]
        
        self.time_per_update = env_cfg["sim"]["dt"]*self.rollout_steps #rollout Sim time before each NN update

        self.action_lambda = train_cfg["action_lambda"]
        self.ac = ActorCritic(  train_cfg["nn_layer_connections"], 
                                self.num_obs, self.num_actions, 
                                train_cfg["orthonal_init"], 
                                train_cfg["action_init_var"],
                                self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.l_rate)

        if(self.checkpoint):
            self.ac.load_state_dict(torch.load(checkpoint))

        self.tb = SummaryWriter(log_dir = self.log_dir + "/tb")

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
        self.next_obs_buf = obs_dict["obs"].clone()
        
        reward_sum_buf = torch.zeros((self.num_envs), device = self.device, dtype=torch.float)
        reward_sum = 0
        episode_reward_sum = 0
        mean_ep_reward = torch.zeros(1)
        finished_episodes = 0
        best_score = 0
        self.global_step = 0
        self.update_step = 0

        while(self.update_step < self.total_updates):
            
            #Collect rollout
            for step in range(self.rollout_steps):
                self.obs_buf[step] = self.next_obs_buf
                self.reset_buf[step] = self.next_reset_buf
                #Only inference, no grad needed
                with torch.no_grad():
                    self.value_buf[step] = self.ac.critic(self.obs_buf[step]).squeeze()
                    action, prob_dist = select_action_multinormaldist(self.ac.actor(self.obs_buf[step]), self.ac.actor_variance)
                self.log_prob_buf[step] = prob_dist.log_prob(action)
                next_obs_dict, reward, next_reset, _ = self.vecenv.step(action)
                
                self.next_obs_buf = next_obs_dict["obs"]
                self.reward_buf[step] = reward
                self.next_reset_buf = next_reset
                self.action_buf[step] = action

                #Accumulate non time-adjusted score
                reward_sum += reward.mean()
                
                #Calculate mean episode reward
                reward_sum_buf += reward
                r_finished = torch.masked_select(reward_sum_buf, next_reset.bool())
                finished_episodes += len(r_finished)
                if len(r_finished) > 0:
                    episode_reward_sum += r_finished.sum()
                reward_sum_buf *= 1-next_reset
                
                #TODO: Normalize reward
                
                self.global_step+=1

            #Calculate generalized advantage estimate, looping backwards
            with torch.no_grad():
                self.value_buf[self.rollout_steps] = self.ac.critic(self.next_obs_buf).squeeze()
            self.reset_buf[self.rollout_steps] = self.next_reset_buf
            gae_buf = torch.zeros((self.rollout_steps, self.num_envs), device=self.device, dtype=torch.float)
            gae_next = 0
            for step in reversed(range(self.rollout_steps)):
                delta = self.reward_buf[step] + self.gamma * self.value_buf[step+1] * (1 - self.reset_buf[step+1]) - self.value_buf[step]
                gae_buf[step] =  delta + self.gamma * self.lambda_ * gae_next * (1 - self.reset_buf[step+1])
                gae_next = gae_buf[step]
            #Value buf is one element longer than gae_buf, we have bootstrapped next_value
            return_buf = gae_buf + self.value_buf[0:self.rollout_steps]

            #Update network
            if(not self.checkpoint):
                shuffled_indicies = np.arange(self.rollout_steps)
                for k in range(self.num_epoch):
                        np.random.shuffle(shuffled_indicies)
                        for mb in range(self.rollout_steps//self.minibatch_size):
                            minibatch_indicies = shuffled_indicies[mb*self.minibatch_size:(mb+1)*self.minibatch_size]
                            _, prob_dists = select_action_multinormaldist(self.ac.actor(self.obs_buf[minibatch_indicies]), self.ac.actor_variance)
                            log_probs_new = prob_dists.log_prob(self.action_buf[minibatch_indicies])

                            values = self.ac.critic(self.obs_buf[minibatch_indicies]).squeeze()
                            log_prob = self.log_prob_buf[minibatch_indicies]
                            returns = return_buf[minibatch_indicies]
                            advantage = gae_buf[minibatch_indicies]
                            self.update_net(log_prob, log_probs_new, values, returns, advantage)
                    
            
            #End of update tasks
            self.ac.actor_variance *= self.action_lambda
            self.tb.add_scalar("Advantage", gae_buf.mean(), self.global_step)
            
            #Get mean episode reward
            if len(r_finished) > 0:
                mean_ep_reward = episode_reward_sum/finished_episodes
            self.tb.add_scalar("Score/episode_reward", float(mean_ep_reward), self.global_step)
            finished_episodes = 0
            episode_reward_sum = 0

            #Calculate score per second
            reward_time_average = reward_sum/(self.time_per_update*self.eval_freq)
            self.tb.add_scalar("Score/time_average_reward", float(reward_time_average), self.global_step)
            reward_sum = 0

            if self.update_step % self.eval_freq == 0:
                score = mean_ep_reward
                print(  "Timestep " + str(self.global_step) + ", Network updates:" + str(self.update_step) + ": Score: " + 
                        str(round(score.item(), 2)) + ", Action Variance: " + str(round(self.ac.actor_variance.data.mean().item(), 2)))
                if score > best_score:
                    print("Saving checkpoint ")
                    best_score = score
                    torch.save(self.ac.state_dict(), self.log_dir + "/" + self.now + "_best_weigth")
            self.update_step += 1

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

        tot_loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        tot_loss.mean().backward()
        #nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
        self.optimizer.step()

        self.tb.add_scalar("Loss/total", tot_loss, self.global_step)
        self.tb.add_scalar("Loss/Actor", actor_loss, self.global_step)
        self.tb.add_scalar("Loss/Critic", critic_loss, self.global_step)
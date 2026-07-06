import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os

# if torch.backends.mps.is_available():
#     device="mps"
# elif torch.cuda.is_available():
#     device="cuda"
# else:
#    device="cpu"

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This model has been configured to run exclusively on an NVIDIA GPU (CUDA). "
        "Please ensure you have installed a PyTorch build with CUDA support."
    )
device = "cuda"



RUNS_DIR="runs"
os.makedirs(RUNS_DIR, exist_ok=True)

class Agent:
    def __init__(self,param_set):
        self.param_set=param_set
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params=all_param_set[param_set]
        
        self.alpha=params["alpha"]
        self.gamma=params["gamma"]

        self.epsilon_init=params["epsilon_init"]
        self.epsilon_decay=params["epsilon_decay"]
        self.epsilon_min=params["epsilon_min"]

        self.replay_memory_size=params["replay_memory_size"]
        self.mini_batch_size=params["mini_batch_size"]

        self.network_sync_rate=params["network_sync_rate"]
        
        self.reward_threshold=params["reward_threshold"]

        self.loss_fn=nn.MSELoss()
        self.optimizer=None

        self.LOG_FILE=os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE=os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        # We disable LIDAR observations (use_lidar=False) to get a concise 12-dimensional
        # state vector, matching our model's default input dimension (state_dim=12).
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)

        num_states=env.observation_space.shape[0]  #inout dim
        num_actions=env.action_space.n      #output dim

        policy_dqn=DQN(num_states, num_actions).to(device) #policy network

        if is_training:
            memory=ReplayMemory(self.replay_memory_size) #Replay Memory
            epsilon=self.epsilon_init

            target_dqn=DQN(num_states, num_actions).to(device) #target network

            #copy the wt and bias vals from policy => target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)

            best_reward=float("-inf")

        #Testing
        else:
            #Loading the best model i.e the best policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))    
            policy_dqn.eval() # Because we dont want to train , we just want to predict and not update the wts and biases
        
        for episode in itertools.count():
            state, _ = env.reset()

            #create tensor for dqn processing
            state=torch.tensor(state, dtype=torch.float, device=device)

            episode_reward=0
            terminated=False
            truncated=False

            # Check both terminated and truncated to prevent running/stepping on finished episodes
            while (not terminated and not truncated and episode_reward<self.reward_threshold):
                # Next action:
                # (feed the observation to your agent here)

                if is_training and random.random()<epsilon:
                    action = env.action_space.sample() #Explore

                    #create tensor for dqn processing
                    action=torch.tensor(action, dtype=torch.long, device=device)
                else:
                    #since we are only picking the optimal and no learning is happening, so we dont want gradients to get computed
                    with torch.no_grad(): 
                        #Neural networks expect a batch of inputs, so unsqueeze(dim=0) adds a batch dimension:,i.e converts 1d into 2d
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() #Exploit

                # Processing: terminated =>Done. We unpack both terminated and truncated
                next_state, reward, terminated, truncated, _ = env.step(action.item())
            
                #create tensor for dqn processing
                reward=torch.tensor(reward, dtype=torch.float, device=device)
                next_state=torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps+=1

                    # Optimize at every step during training instead of once per episode
                    if len(memory)> self.mini_batch_size:
                        #get sample experiences
                        mini_batch=memory.sample(self.mini_batch_size)
                        
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        #sync the target network when steps reach the sync rate
                        if steps>=self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps=0

                #Update state
                state=next_state
                episode_reward += reward.item()

            if is_training:
                print(f"Episode : {episode+1} | Reward: {episode_reward} | Epsilon: {epsilon}")
            else:
                print(f"Episode : {episode+1} | Reward: {episode_reward}")


            if is_training:
                #Epsilon Decay
                epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)

                if episode_reward>best_reward:
                    log_msg=f"best reward : {episode_reward} for episode={episode+1}"

                    # save log file and best model 
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg+"\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward=episode_reward

                    print(f"Saved new best model: Reward = {best_reward}")

            # env.close() we have to indefinitely run / manually stop
        
        
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        # get experience

        #training as a batch -> faster
        states, actions, next_states, rewards, terminations=zip(*mini_batch)
        
        states=torch.stack(states)
        actions=torch.stack(actions)
        next_states=torch.stack(next_states)
        rewards=torch.stack(rewards)
        terminations=torch.tensor(terminations).float().to(device)

        # Calculate target Q-Values- if terminations=true =>zero
        with torch.no_grad():
            target_q=rewards+(1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        #calculate y_pred  i,e Q-value from current policy
        current_q=policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Calculate Loss
        loss=self.loss_fn(current_q, target_q)

        #Model Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # training each experience -> slower 

        # for state, action, next_state, reward, terminated in mini_batch:

        #     if terminated:
        #         target_q=reward
        #     else:
        #         with torch.no_grad():
        #             target_q=reward + self.gamma * target_dqn(next_state).max()
            
        #     current_q=policy_dqn(state)

        #     #loss
        #     loss=self.loss_fn(current_q, target_q)

        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)


            
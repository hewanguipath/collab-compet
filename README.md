[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[image3]: https://openai.com/content/images/2017/06/nipsdiagram_2.gif
[image4]: https://nervanasystems.github.io/coach/_images/ddpg.png
[image5]: record.png

# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  


##### 1.	Define the Network:
Structure: MA-DDPG
I use 2 DDPG Agent, Delayed Deep Deterministic policy gradient in this practice. And share their “predict actions next”, and “predict actions”

![DDPG][image4]

Network size:
<p> As the Vector Observation space size in this environment is 24 each, 24 x 2, action space size is 2 each, 2 x 2
<p> So the Actor input as full status (1, 48), and output as full actions (1, 4), Critic input as full status (1, 48) and full action (1, 4), output as 1, the value
<p> I have defined First hidden layer units as 256 and 2nd hidden layer as 128 for Actor, small enough to run in CPU with the balance of simplicity and efficiency.
<p> For the Critic First hidden layer units as 256 and 2nd hidden layer as 128, 

##### 2.	Define the Replay Buffer:
<p> Since the 2 agents observed different status and take different actions, so the reply buffer must be separated, so I linked 2 reply buffers for each. 
<p> And I used a prioritized reply buffer, but not sum tree, takes N time rather than Log N to get random selections with prioritize and distribution. Hence, the size of the buffer should not be too long, so I take 1e5.
<p> The buffer would be filled full around 600 episode, and then put with new episodes, which has better performance trajectories
<p> In each step of the episodes, push the full states, concoction of status of both agents, full action, full next status, individual reward and done to the buffer.

##### 3.	Define the Agent:
<p> 2 Levels of the Agent, DDPG agent, focus on basic DDPG algorithm, and a MADDPG agent which will coordinate data share between 2 DDPG agent.

![MADDPG][image3]

<p> Each Agent takes full status and full actions as input, actors will generate the Individual actions, and concatenate the generated action of the other agent, as input to the critic.
<p> Exploration, I did not use epsilon-greedy, I have a random noise as the class example

##### 4.	Training the DDQN:
<p> Due to the Network is bigger than 2 projects before, and share middle level info, requires additional feed forward calculation. Furthermore, priority reply buffer needs one more feed forward, and random sample with priority distribution. So the training takes much more time than previous projects. 

<p> Max_Steps: 1001 as observed in the env, it will be finished in 1001
<p> Skip_timesteps, 100, since very rare hit ball in the very begging. 
<p> Random action generation, random normal distribution
<p> Batch_size, 128 is sufficient , 256 would be too slow.
<p> it converges quite slow at very begining, but would be very fast at last.
    
```

Episode 50	Average Score: 0.000	Max Score: 0.00
Episode 100	Average Score: 0.019	Max Score: 0.10
Episode 150	Average Score: 0.030	Max Score: 0.10
Episode 200	Average Score: 0.010	Max Score: 0.10
Episode 250	Average Score: 0.040	Max Score: 0.10
Episode 300	Average Score: 0.010	Max Score: 0.10
Episode 350	Average Score: 0.000	Max Score: 0.00
Episode 400	Average Score: 0.020	Max Score: 0.10
Episode 450	Average Score: 0.020	Max Score: 0.20
Episode 500	Average Score: 0.000	Max Score: 0.00
Episode 550	Average Score: 0.059	Max Score: 0.10
Episode 600	Average Score: 0.077	Max Score: 0.20
Episode 650	Average Score: 0.078	Max Score: 0.20
Episode 700	Average Score: 0.078	Max Score: 0.10
Episode 750	Average Score: 0.120	Max Score: 0.20
Episode 800	Average Score: 0.118	Max Score: 0.20
Episode 850	Average Score: 0.120	Max Score: 0.20
Episode 900	Average Score: 0.130	Max Score: 0.400
Episode 950	Average Score: 0.190	Max Score: 0.400
Episode 1000	Average Score: 0.170	Max Score: 0.40
Episode 1050	Average Score: 0.119	Max Score: 0.39
Episode 1100	Average Score: 0.200	Max Score: 0.400
Episode 1150	Average Score: 0.520	Max Score: 1.10
Episode 1200	Average Score: 0.310	Max Score: 0.700
Episode 1250	Average Score: 0.880	Max Score: 1.300
Finished at Episode 1256	Reach Average Score: 1.078!

```
![DDPG][image5]
    
##### 5.	Interesting observations:
    <p> For sharing the intermedia information among agents, I see someone has done alternatively [3], rather than sharing the predicted action and next actions with actor_target and actor_local, he arbitrarily shared the actual action of the other agent as the predict action and next action as input to the critic.
    <p> With this way, the calculation would definitely less, according to his training record, it converges faster, but I can’t reproduce it, I had try his method, with switch “SHARE_ACTUAL_ACTION” in my code, seems very hard to converge at score around 0.3.

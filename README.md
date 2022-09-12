# CabDriverRetention_RL
Reinforcement Learning

"Problem Statement:
At SuperCabs, a leading app-based cab provider in a large Indian metro city. In this highly competitive industry, retention of good cab drivers is a crucial business driver, and you believe that a sound RL-based system for assisting cab drivers can potentially retain and attract new cab drivers. Cab drivers, like most people, are incentivised by a healthy growth in income. The goal of your project is to build an RL-based algorithm which can help cab drivers maximise their profits by improving their decision-making process on the field.

The Need for Choosing the ""Right"" Requests:
Most drivers get a healthy number of ride requests from customers throughout the day. But with the recent hikes in electricity prices (all cabs are electric), many drivers complain that although their revenues are gradually increasing, their profits are almost flat. Thus, it is important that drivers choose the 'right' rides, i.e. choose the rides which are likely to maximise the total profit earned by the driver that day. 

For example, say a driver gets three ride requests at 5 PM. The first one is a long-distance ride guaranteeing high fare, but it will take him to a location which is unlikely to get him another ride for the next few hours. The second one ends in a better location, but it requires him to take a slight detour to pick the customer up, adding to fuel costs. Perhaps the best choice is to choose the third one, which although is medium-distance, it will likely get him another ride subsequently and avoid most of the traffic. 

There are some basic rules governing the ride-allocation system. If the cab is already in use, then the driver won’t get any requests. Otherwise, he may get multiple request(s). He can either decide to take any one of these requests or can go ‘offline’, i.e., not accept any request at all. 

Markov Decision Process:
Taking long-term profit as the goal, you propose a method based on reinforcement learning to optimize taxi driving strategies for profit maximization. This optimisation problem is formulated as a Markov Decision Process.

You can check the MDP from below. The major simplifying assumptions are also mentioned.

In this project, you need to create the environment and an RL agent that learns to choose the best request. You need to train your agent using vanilla Deep Q-learning (DQN) only and NOT a double DQN. You have learnt about the two architectures of DQN (shown below) - you are free to choose any of these."

# "Goals:
1. Create the environment: You are given the ‘Env.py’ file with the basic code structure. This is the ""environment class"" - each method (function) of the class has a specific purpose. Please read the comments around each method carefully to understand what it is designed to do. Using this framework is not compulsory, you can create your own framework and functions as well.

2. Build an agent that learns to pick the best request using DQN. You can choose the hyperparameters (epsilon (decay rate), learning-rate, discount factor etc.) of your choice.

    2.1. Training depends purely on the epsilon-function you choose. If the ?decays fast, it won’t let your model explore much and the Q-values will converge early but to suboptimal values. If ?decays slowly, your model will converge slowly. We recommend that you try converging the Q-values in 4-6 hrs.  We’ve   created a sample ?-decay function at the end of the Agent file (Jupyter notebook) that will converge your Q-values in ~5 hrs. Try building a similar one for your Q-network.

    2.2. In the Agent file, we’ve provided the code skeleton. Using this structure is not necessary though.

           You have to submit your final DQN model as a pickle file as well.

3. Convergence- You need to converge your results. The Q-values may be suboptimal since the agent won't be able to explore much in 5-6 hours of simulation. But it is important that your Q-values converge. There are two ways to check the convergence of the DQN model:

    3.1. Sample a few state-action pairs and plot their Q-values along episodes

    3.2. Check whether the total rewards earned per episode are showing stability

Showing one of these convergence plots will suffice.


Important points to consider while training:

1. Choose ?-decay function carefully. Make sure that the decay rate allows the agent to explore the state space maximally at the start, and then to settle down to a fixed exploration rate (the results will converge as soon as ε becomes 0, though results may not be optimal if the agent doesn’t explore much).

2. List down all the metrics (such as total rewards for n episodes, loss values, q-values, etc.) you want to track for checking the convergence and save them after every few episodes. It is recommended that you store and check your results after every ~1000 episodes.

3. Don’t forget to save your model (weights) after every few iterations.

4. Make sure to debug your code before running for a large number of episodes. Run for 3-4 steps in an episode, and check whether the reward, next state computations etc. are correct. Only when the code is debugged, run it for a larger number of episodes.

5. For a Windows 64-bit system, and with an 8GB RAM, it takes around 1 hour to train for 1000 episodes with the average length of episodes 100."

# MDP Formulation:
"Objective:
The objective of the problem is to maximise the profit earned over the long-term.

Decision Epochs:
The decisions are made at an hourly interval; thus, the decision epochs are discrete.

Assumptions:
1. The taxis are electric cars. It can run for 30 days non-stop, i.e., 24*30 hrs. Then it needs to recharge itself. If the cab driver is completing his trip at that time, he’ll finish that trip and then stop for recharging. So, the terminal state is independent of the number of rides covered in a month, it is achieved as soon as the cab driver crosses 24*30 hours.
2. There are only 5 locations in the city where the cab can operate.
3. All decisions are made at hourly intervals. We won’t consider minutes and seconds for this project. So for example, the cab driver gets requests at 1:00 pm, then at 2:00 pm, then at 3:00 pm and so on. So, he can decide to pick among the requests only at these times. A request cannot come at (say) 2.30 pm.
4. The time taken to travel from one place to another is considered in integer hours (only) and is dependent on the traffic. Also, the traffic is dependent on the hour of the day and the day of the week.

State:
The state space is defined by the driver’s current location along with the time components (hour of the day and the day of the week). A state is defined by three variables:
𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘 𝑤ℎ𝑒𝑟𝑒 𝑖 = 0 … 𝑚 − 1; 𝑗 = 0 … . 𝑡 − 1; 𝑘 = 0 … . . 𝑑 − 1

Where 𝑋𝑖 represents a driver’s current location, 𝑇𝑗 represents time component (more specifically hour of the day), 𝐷𝑘 represents the day of the week
• Number of locations: m = 5
• Number of hours: t = 24
• Number of days: d = 7
A terminal state is achieved when the cab completes his 30 days, i.e., an episode is 30 days long."
"Actions:
Every hour, ride requests come from customers in the form of (pick-up, drop) location. Based on the current ‘state’, he needs to take an action that could maximise his monthly revenue. If some passenger is already on board, then the driver won’t get any requests.

Therefore, an action is represented by the tuple (pick-up, drop) location. In a general scenario, the number of requests the cab driver can get at any state is not the same. We can model the number of requests as follows:

The number of requests (possible actions) at a state is dependent on the location. Say, at location A, you get 2 requests on average and at location B, you get 12 requests on average. That means when at A, the cab driver is likely to get 2 customer requests from anywhere to anywhere of the form (𝑝, 𝑞).

For each location, you can sample the number of requests from a Poisson distribution using the mean λ defined for each location as below:

Location     λ (of Poisson Distribution)
Location A      2
Location B     12
Location C     4
Location D     7
Location E     8

The upper limit on these customers’ requests (𝑝, 𝑞) is 15.

Apart from these requests, the driver always has the option to go ‘offline’ (accept no ride). The noride action just moves the time component by 1 hour. So, you need to append (0,0) action to the customer requests.

There’ll never be requests of the sort where pickup and drop locations are the same. So, the action space A will be: (𝑚 − 1) ∗ 𝑚 + 1 for m locations. Each action will be a tuple of size 2. You can define action space as below:
• pick up and drop locations (𝑝, 𝑞) where p and q both take a value between 1 and m;
• (0, 0) tuple that represents ’no-ride’ action.

For example, if the set of all possible locations is of size 3: {A, B, C}. So, at state (A, 6:00 pm, Wednesday), his possible actions would be of the form: (𝑖, 𝑗) where i and j can be any location from {A, B, C}, but i ≠ j. The following table shows all possible actions the driver can take.
(A, B)       (C, A)
(A, C)       (C, B)
(B, C)       (B, A)
(0, 0)

The average number of requests received when the driver is in location A is 2. So, the cab driver will get any two random requests from the above table, say (A, B) and (C, A). Also, he’ll always have the option of ‘no-ride’. So, his possible actions at that state would be:
(𝐴, 𝐵), (𝐶, 𝐴), (0, 0)

His action should be to pick the best request among these three"
"State Transition:
Given the current state 𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘, the next state 𝑠’ will be as following:
𝑠′ =
𝑋𝑞𝑇𝑡′𝐷𝑑′𝑎 = (𝑝, 𝑞)
𝑋𝑝𝑇𝑡′𝐷𝑑′𝑎 = (0,0)

Where 𝑡′, 𝑑′ represents the time and day respectively after taking an action.

You can calculate the total time taken to reach from one point to other from the Time Matrix (calculated basis the historical data) provided to you in the zip file. You don't need to learn a distribution of the time taken; all possible values of (pick up, drop, t, d) are available in the ‘TM.npy’ file, you simply need to look them up.

Time Matrix is a 4-D matrix. The 4 dimensions are as below:
• Start location
• End location
• Time of the day
• Day of the week

Python indices for these dimensions are as:
𝑇𝑖𝑚𝑒 − 𝑚𝑎𝑡𝑟𝑖𝑥[𝑠𝑡𝑎𝑟𝑡 − 𝑙𝑜𝑐][𝑒𝑛𝑑 − 𝑙𝑜𝑐][ℎ𝑜𝑢𝑟 𝑜𝑓 𝑡ℎ𝑒 𝑑𝑎𝑦] [𝑑𝑎𝑦 𝑜𝑓 𝑡ℎ𝑒 𝑤𝑒𝑒𝑘]

This matrix has been calculated considering the distance between two locations and traffic conditions, which generally depends on the hour of the day and the-day of the week. To make the problem manageable, we divided the 24-hour frame into 4 segments:
from 12:00 am to 6:00 am, 6:00 am to 12:00 pm, 12:00 pm to 6:00 pm and 6:00 pm to 12:00 am.

You are given this time-matrix in the zip file shared. (Please run it once to understand its dimensions).

Reward:
Your objective is to maximize the profit of a driver. Let 𝐶𝑓 be the amount of battery consumed per hour and 𝑅𝑘 be the revenue he obtains from the customer for every hour of the ride. The values of these parameters are defined in the skeleton code provided to you. Also, we have assumed that both the cost and the revenue are purely functions of time, i.e. for every hour of driving, the cost (of battery and other costs) and the revenue (from the customer) is the same - irrespective of the traffic conditions, speed of the car etc.

So, the reward function will be (revenue earned from pickup point 𝑝 to drop point 𝑞) - (Cost of battery used in moving from pickup point 𝑝 to drop point 𝑞) - (Cost of battery used in moving from current point 𝑖 to pick-up point 𝑝). Mathematically,

𝑅(𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘) =            𝑅𝑘 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞)) − 𝐶𝑓 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞) + 𝑇𝑖𝑚𝑒(𝑖, 𝑝))     𝑎 = (𝑝, 𝑞)
                                    - 𝐶𝑓                                                                           𝑎 = (0,0)

Where 𝑋𝑖
represents a driver’s current location, 𝑇𝑗 represents time component (more specifically hour of the day), 𝐷𝑘 represents the day of the week, 𝑝 represents the pickup location and 𝑞 represents the drop location."

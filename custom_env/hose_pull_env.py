
''' 
__author__ = "Dom McKean Loughborough University"
__credits__ = ["OpenAi Gym", "Oleg Klimov"]
__version__ = "0.0.1"
__maintainer__ = "Dom McKean"
__github_username__ = "Porkpy"
'''

''' In this environment, the robot position is also the hose position; therefore, the force
calculator can be invoked on the agent's action output which acts similarly to a critic. 
The force calculator model returns the predicted force for the current action which is 
then used to reward or punish the agent.
This is in contrast to how the MPC style controller used the force calculator, 
where the controller would suggest an action to the force calculator, the calculator
would return whether the force was above or below the maximum specified force, which would
in turn, if necessary, invoke the 'beam adjust' controller which iteratively adjusted the 
suggested action by reducing the bend angle and lengthening the pull length until the 
action falls within parameters. In this RL environment, there is no forward planning. 
The agent will need to learn on-line, using the force calculator as a simulation
of the real world dynamics.        
'''




import sys, os
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np 
import contact as c
import scipy.stats as stats
import pos_calc as pc 
import os
import sys
import pickle
import matplotlib.pyplot as plt
from pylab import * 
from prettytable import PrettyTable


# file handling. This section removes previous pickle files.
directory = os.getcwd()
  #print(os.getcwd())
file_path = "/home/dom/Desktop/mpc"
if directory == file_path:
  for file in glob.glob("*.pickle"):
    os.remove(file)
elif os.path.exists(file_path):
  os.chdir(file_path)
else:
  f = open(file_path)
os.chdir(file_path)


max_force = 5



# The defined observation_space is the input to the network/agent and the defined action_space is the output from the network/agent.


def get_force(angle, length):
  force = c.force_calc(angle, length)
  return force


# Initial robot position. Set to x=0, y=0.
def get_robot_postion():
  robot_start_position = (0., -0.5)
  return robot_start_position[0], robot_start_position[1]

# Set the random goal position in the task space. Logic uses a Gaussian norm with limits.
def goal_pos():

  x_lower, x_upper = 68, 300 # the lower and upper limits of the task space 
  x_mu, x_sigma = 250, 50 # mean and standard deviation 
  x = stats.truncnorm((x_lower - x_mu) / x_sigma, (x_upper - x_mu) / x_sigma, loc=x_mu, scale=x_sigma) 
  x = x.rvs()
  x /= 1_000 
  #print(x)

  y_lower, y_upper = -766, -466 # the lower and upper limits of the task space 
  y_mu, y_sigma = -700, 50 # mean and standard deviation 
  y = stats.truncnorm((y_lower - y_mu) / y_sigma, (y_upper - y_mu) / y_sigma, loc=y_mu, scale=y_sigma) 
  y = y.rvs()
  y /= 1_000 
  #print(y)
  return x, y



# Instantiate the Gym environment constructor. 

class HosePullEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {
        'render.modes': ['human']
  }


  def __init__(self):
    super(HosePullEnv, self).__init__()
    

    self.action_space = spaces.Dict(dict(
      x=spaces.Box(0, 0.01, shape=(1,), dtype='float32'), # choose an angle action between 0 and 89 degrees.
      y=spaces.Box(-0.01, 0., shape=(1,), dtype='float32'), # choose a move length action between 0mm and 49mm.
      ))

    self.observation_space = spaces.Dict(dict(
      x=spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float32'), # Observe the agent's pull angle output in degrees.
      y=spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float32'), # Observe the agent's pull length output in mm.
      force=spaces.Box(-np.inf, np.inf, shape=(1,), dtype='float32'), # Observe the force calculated by the force_calculator.
      )) 
    
    
    #print("observation space:",  self.observation_space)

    
    # Use the same start position for the robot = (0,0) which is the orifice position.

    # Create a random goal position in the task space. 
    # the random positions are taken from a Gaussian distribution so as to have a little control 
    # on where the random positions are likely to appear. This reduces the probability of the  
    # robot position being further from the orifice than the goal.         
  




  def step(self, action):

    """Run one time-step of the environment's dynamics. When end of
    episode is reached, you are responsible for calling `reset()`
    to reset this environment's state.
    Accepts an action and returns a tuple (observation, reward, done, info).
    Args:
      action (object): an action provided by the agent
    Returns:
      observation (object): agent's observation of the current environment
      reward (float) : amount of reward returned after previous action
      done (bool): whether the episode has ended, in which case further step() calls will return undefined results
      info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """
  #   Moves the environment forward by one step. Requires an action as input and returns a new observation.
  #   Also included are the reward, movement dynamics management, calculation of status, controls for 
  #   completing the episode.

    # The action is returned by the agent and we can make decisions based on those actions. 

    
    x = action['x'] # Get x and y output from dictionary. 
    y = action['y']
    

    observation = [] # Clear previous observations. 
    
    with open('state.pickle', 'rb') as file1: # If file exists, load file.
      previous_x, previous_y = pickle.load(file1)
    print("previous_x = ", previous_x, "previous_y = ", previous_y)

    with open('goal.pickle', 'rb') as file2: # If file exists, load file.
        goal_x, goal_y = pickle.load(file2)
 

    print("x=", x, "y =", y)

    x += previous_x
    y += previous_y

    # Convert new x,y back into angle and length.

    # Termination and reward shaping.
    done = False

    if x > 1 or x < -1: # Robot out of bounds :(
      done = True
      reward = -100       

    if y < -1 or y > 1: # Robot out of bounds :(
      done = True
      reward = -95

    x_error = goal_x - x # Determine the amount of error between current position and goal. 
    y_error = goal_y - y
    print("x = ", x, "goal_x =", goal_x, "previous_x = ", previous_x)
    print("x_error = ", x_error, "y_error = ", y_error)
    print("move length", x- previous_x, y - previous_y)


    print("x = ", x, "y = ", y, "before angle length calc")
    angle, length = pc.angle_length_calc(x,y) # Convert agent's  x y coordinate actions into angle and length in the task space.
    print("angle = ", angle, "length = ", length, "after angle length calc")

    force = get_force(angle, length) # Get force prediction based upon angle and length. 
    #print(force)


    if np.abs(x_error) < 0.1 and np.abs(y_error) < 0.1: # Goal reached, yay!
      done = True
      reward = 100 #- force
      print("Goal = ", goal_x, goal_y)
      print("Goal Reached!")

    if np.abs(x_error) < 0.2 and np.abs(y_error) < 0.2: # Goal reached, yay!
      done = False
      reward = 50 #- force
      print("Goal = ", goal_x, goal_y)
      print("Goal Reached!")
    
    
  
    # if force > max_force: # Force too high :(
    #   done = True
    #   reward = -81
    # print("force = ",force, "Newtons")

    if done == False: # Continue to next step in episode.
      reward = -1 #- force

    with open('reward.pickle', 'rb') as file3: # If file exists, load file.
        previous_reward = pickle.load(file3)

    reward += previous_reward

    pickel_file = reward
    with open('reward.pickle', "wb") as file3:
      pickle.dump(pickel_file, file3, -1)

    #print("reward = ", reward)
    # convert angle and length into Cartesian coordinates and serialise to keep track of robot position. 

    print("goal position = ", goal_x, goal_y)

    pickel_file = x, y

    with open('state.pickle', "wb") as file1:
      pickle.dump(pickel_file, file1, -1)

    observation = [goal_x, goal_y, previous_x, previous_y, x, y, force]

    info = {}
    #print("Observation =",observation, "reward =",reward, "Done =",done, info)

    # observation = current position, previous position, goal, force.


    # plt.ion()
    # fig = plt.figure()
    # data = np.array(x,y)
    # plt.title('Task Frame Moves')
    # plt.xlabel('x')
    # plt.ylabel('y') 
    # axes = plt.gca()
    # axes.set_xlim([-1.5,1.5])
    # axes.set_ylim([-1.5,0])
    # x, y = data.T
    # plt.scatter(x,y)
    # plt.show()
    
    plt.ion() 
    axes = plt.gca()
    axes.set_xlim([-0.1,1.0])
    axes.set_ylim([-1.,0.1])
    plt.title('Task Frame Moves')
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.plot(goal_x, goal_y)
    plt.plot(x,y)
    plt.scatter(x,y)
    plt.scatter(goal_x, goal_y)
    #plt.savefig("/home/dom/Desktop/mpc/github_mpc_6.1.20.png")
    plt.pause(0.0001)
    plt.clf()
    

    t = PrettyTable(['Reward', 'X Error', 'Y Error', 'Force' ])
    t.add_row([reward, x_error, y_error, force])
    print(t)
    
    return np.array(observation), reward, done, info 

  def reset(self):

    """Resets the state of the environment and returns an initial observation.
    Returns:
    observation (object): the initial observation.
    """  
    
    reward = 0 # instantiate a pickle file that can be called by step function. 
    pickel_file = reward
    with open('reward.pickle', "wb") as file3:
      pickle.dump(pickel_file, file3, -1)
    
    goal_x, goal_y = goal_pos() # get new random goal position.

    pickel_file = goal_x, goal_y
    with open('goal.pickle', "wb") as file2:
      pickle.dump(pickel_file, file2, -1)

    x, y = 0., -0.5 #get_robot_postion() # get initial robot position. Always at home (0,0) for now.
    force = 0
    previous_x = 0
    previous_y = -0.5
    pickel_file = previous_x, previous_y
    with open('state.pickle', "wb") as file1:
      pickle.dump(pickel_file, file1, -1)
    print("")
    print(x, y)
    print("")
    state = np.array([goal_x, goal_y, previous_x, previous_y, x, y, force]) # Build an np.array of observations. 
    #print("State = ",state)

    return np.array(state).astype(np.float32) # returns an np.array of the initial observations.




  

if __name__ == '__main__':
  pass


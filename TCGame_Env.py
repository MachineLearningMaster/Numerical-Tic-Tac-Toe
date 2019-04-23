from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product
import copy



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] 

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        if (sum(curr_state[0:3:1])==15 or sum(curr_state[3:6:1])==15 or sum(curr_state[6:9:1])==15 or
            sum(curr_state[0:9:3])==15 or sum(curr_state[1:9:3])==15 or sum(curr_state[2:9:3])==15 or
            sum(curr_state[0:9:4])==15 or sum(curr_state[2:8:2])==15):
            return True
 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        state_next=copy.copy(curr_state)
        state_next[curr_action[0]]=curr_action[1]
        return state_next


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. 
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        obs_1=self.state_transition(curr_state,curr_action)
        is_terminal,str_state=self.is_terminal(obs_1)
        
        if is_terminal==True:
            if str_state=='Win':reward=10
            elif str_state=='Tie':reward=0
            return obs_1,reward,is_terminal
        
        else:
            valid_actions_env=[action for action in self.action_space(obs_1)[1]]
            curr_action_env=random.sample(valid_actions_env,1)[0]
            obs_2=self.state_transition(obs_1,curr_action_env)
            
            is_terminal,str_state=self.is_terminal(obs_2)
            if is_terminal==True:
                if str_state=='Win':reward=-10
                elif str_state=='Tie':reward=0
                return obs_2,reward,is_terminal
            else:
                return obs_2,-1,is_terminal
        

    def reset(self):
        return self.state

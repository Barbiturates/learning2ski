# -*- coding: utf-8 -*-
import logging
import pickle


log = logging.getLogger(name=__name__)

POSSIBLE_ACTIONS = {'noop': 0, 'left': 1, 'right': 2}


class Agent(object):
    """docstring for Agent"""

    def __init__(self):
        super(Agent, self).__init__()
        self.possible_actions = POSSIBLE_ACTIONS
        self.n_actions = len(self.possible_actions)

    def start_state(self):
        """
        The state this agent is in before the episdoe starts

        Returns
        -------
        None
        """
        return None

    def actions(self, state=None):
        return range(self.n_actions)

    def act(self, state, *args, **kwargs):
        """
        Return an action based on the last observed state

        Parameters
        ----------
        state : tuple of:
            - numpy array (250, 160, 3)
                An RGB image of the game board
            - anything

        Returns
        -------
        int
            An action from self.actions
        new_state : anything
            The second part of the updated state info
        """
        raise NotImplementedError('Override me!')

    def react(self, state, action, reward, done, new_state, *args, **kwargs):
        """
        Incorporate feedback from simulation

        Parameters
        ----------
        state : tuple of:
            - numpy array (250, 160, 3)
                An RGB image of the game board
            - anything
        action : int
            The action that was taken
        reward : int
            The reward that was given
        done : bool
            Whether this ends the episode
        new_state : tuple of:
            - numpy array (250, 160, 3)
                An RGB image of the game board
            - anything
        """
        raise NotImplementedError('Override me!')

    def save(self, path):
        """
        Save the Agent (after learning)

        Parameters
        ----------
        path : str
            Where to save the agent
        """
        with open(path, mode='w') as fout:
            pickle.dump(self, fout)

    @classmethod
    def load(path):
        """
        Load an Agent from a path

        Parameters
        ----------
        path : str
            Where to load the agent from

        Returns
        -------
        Agent
            The saved agent
        """
        with open(path, mode='r') as fin:
            return pickle.load(fin)

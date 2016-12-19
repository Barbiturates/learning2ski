# -*- coding: utf-8 -*-
import logging

import numpy as np

import agents.base


log = logging.getLogger(name=__name__)


class Random(agents.base.Agent):
    """docstring for Random"""
    def __init__(self):
        super(Random, self).__init__()

    def act(self, state, *args, **kwargs):
        """
        Return a random action

        Parameters
        ----------
        state : numpy array (250, 160)
            A green-channel image of the game board

        Returns
        -------
        int
            An action from self.actions
        """
        return np.random.randint(self.n_actions), None

    def react(self, state, action, reward, done, new_state, *args, **kwargs):
        """
        Ignore feedback from simulation

        Parameters
        ----------
        state : numpy array (250, 160)
            A green-channel image of the game board
        action : int
            The action that was taken
        reward : int
            The reward that was given
        done : bool
            Whether this ends the episode
        new_state : numpy array (250, 160)
            A green-channel image of the game board
        """
        pass

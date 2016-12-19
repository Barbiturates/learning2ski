# -*- coding: utf-8 -*-
import logging

import agents.base


log = logging.getLogger(name=__name__)


class Straight(agents.base.Agent):
    """Go straight down the hill"""
    def __init__(self):
        super(Straight, self).__init__()

    def act(self, state, *args, **kwargs):
        """
        Always choose 'noop'

        Parameters
        ----------
        state : numpy array (250, 160)
            A green-channel image of the game board

        Returns
        -------
        int
            An action from self.actions
        """
        return 0

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

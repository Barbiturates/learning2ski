# -*- coding: utf-8 -*-
import logging

import readchar
import readchar.key

import agents.base


log = logging.getLogger(name=__name__)


class Human(agents.base.Agent):
    """Let a human steer"""
    def __init__(self):
        super(Human, self).__init__()

    def act(self, state, *args, **kwargs):
        """
        Waits for the user to select an action

        Parameters
        ----------
        state : numpy array (250, 160)
            A green-channel image of the game board

        Returns
        -------
        int
            An action from self.actions
        """
        cmap = {
            'a': self.possible_actions['right'],
            's': self.possible_actions['noop'],
            'd': self.possible_actions['left'],
            readchar.key.CTRL_C: readchar.key.CTRL_C,
        }

        c = None
        while c not in cmap.keys():
            c = readchar.readkey()

        if c == readchar.key.CTRL_C:
            raise ValueError('Stopping!')

        return cmap[c]

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

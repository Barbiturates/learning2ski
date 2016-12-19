# -*- coding: utf-8 -*-
import collections
import logging

import numpy as np
import sklearn.linear_model
import sklearn.preprocessing

import agents.base
import features


log = logging.getLogger(name=__name__)


class Learning2Ski(agents.base.Agent):
    """
    Fitted Q-Iteration + Experience Replay + Pseudo Rewards + Hand-engineered
    features + Q-function approximation
    """
    def __init__(self,
                 learning=True,
                 n_history=5,
                 discount=0.99,
                 iteration_size=75,
                 batch_size=3000):
        super(Learning2Ski, self).__init__()

        self.discount = discount

        # use this to track how long it's been since we created fake costs
        # and trained
        self.fake_episode_length = 0

        # use this to track how far into the episode we are because we
        # don't want history features to cross real episodes
        self.real_episode_length = 0

        self.n_fake_episodes = 0.0

        self.n_trainings = 0

        # how far back in history to go when creating features
        self.n_history = n_history

        self.iteration_size = iteration_size
        self.batch_size = batch_size
        log.debug('Batch size: {}'.format(self.batch_size))
        self.sars = collections.deque(maxlen=batch_size)

        self.chosen_actions = []

        # metadata that is useful for calculating features, etc., but
        # are not good as features themselves
        self.states = []

        # are we training the network?
        self.learning = learning

        feature_size = len(self.get_features(
            collections.defaultdict(float), 0
        ).ravel())

        self.ridge = sklearn.linear_model.SGDRegressor(warm_start=True)
        self.ridge.t_ = None
        self.ridge.coef_ = np.random.rand(feature_size) - 0.5
        self.ridge.intercept_ = np.random.rand(1) - 0.5

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(self.get_features(
            collections.defaultdict(float), 0
        ))

        # keep track of the score states we've seen
        self.seen_scores = []

    def game_score_state(self, image):
        """
        The score section of the image can be one of at most 20 things:
        20, 19, 18, etc.

        Determine which state it is in. We don't care about the actual number
        just that state_i != state_i+1

        Parameters
        ----------
        image : numpy array (250, 160)
            An green-channel image of the game board

        Returns
        -------
        int
            The index of the score in the seen_scores list
        """
        # get just the score part of the image
        score = image[30:39, 65:82]

        # have we seen this score before?
        for i, seen_score in enumerate(self.seen_scores):
            # if we've seen it
            if np.all(score == seen_score):
                # return its index
                return i

        # if not, keep track of it and return
        self.seen_scores.append(score)
        return len(self.seen_scores) - 1

    def make_metadata(self, image, centiseconds):
        """
        Create the state that we track from the green-channel image of the game

        Parameters
        ----------
        image : numpy array (250, 160)
            A green channel only image of the playing screen
        centiseconds : int
            How much time elapsed between the last state and this one

        Returns
        -------
        dict
            Our state representation
        """
        m = {}
        centiseconds = float(centiseconds)
        m['centiseconds'] = centiseconds
        m['game_score_state'] = self.game_score_state(image)

        # get the skier location
        try:
            skier = np.array(
                features.skier_loc(features.skier_box(image))
            )
        except:
            # Skier is probably behind a tree
            if len(self.states) > 0:
                skier = self.states[-1]['skier']
            else:
                skier = np.array([
                    features.WINDOW_WIDTH / 2.0,
                    (
                        features.SKIER_Y[1] - features.SKIER_Y[0] / 2.0
                    ) + features.SKIER_Y[0]
                ])
        m['skier'] = skier

        # get the nearest flag location
        flag = features.flag_loc(image, skier[1])

        m['flag'] = flag

        m['y_distance_to_flag'] = flag[1] - skier[1]
        m['x_distance_to_flag'] = np.abs(skier[0] - flag[0])
        m['distance_to_flag'] = np.linalg.norm(flag - skier)

        # use this slice to track the speed in y
        # looks like (y, array)
        m['y_delta_slice'] = features.y_delta_slice(image)

        # there is a set of things that we are interested in knowing about
        # the difference between this state and last
        if self.real_episode_length == 0:
            # pixel diffs init to 0
            m['delta_x'] = 0
            m['delta_y'] = 0
            m['score_changed'] = False

            # if we can't get that info, just return
            return m

        m['score_changed'] = (
            m['game_score_state'] != self.states[-1]['game_score_state']
        )

        # get the pixel difference in x between this frame and the last
        m['delta_x'] = skier[0] - self.states[-1]['skier'][0]

        # get the pixel difference in y between this frame and the last
        m['delta_y'] = features.delta_y(
            image, self.states[-1]['y_delta_slice']
        )
        # and then there is a set of things we're interested in knowing about
        # the difference between this state and each previous state up to
        # n_history
        delta_x, delta_y = m['delta_x'], m['delta_y']

        cum_centiseconds = centiseconds

        for i in range(1, self.n_history):
            if len(self.states) < i:
                break

            # speed in pixels per centisecond
            m['speed_x_{}'.format(i)] = delta_x / cum_centiseconds

            # speed in pixels per centisecond
            m['speed_y_{}'.format(i)] = delta_y / cum_centiseconds

            # am i moving downward?
            right = np.array([1, 0])
            moving = np.array(
                [m['speed_x_{}'.format(i)], m['speed_y_{}'.format(i)]]
            )
            norm_moving = np.linalg.norm(moving)
            cos = np.dot(right, moving) / norm_moving

            # if nan (or, surprisingly, 0 -- to handle sliding down the wall)
            # TODO: fix this
            if np.isnan(cos) or cos == 0:
                cos = self.states[-1].get('cos_movement_{}'.format(i), 0)

            m['cos_movement_{}'.format(i)] = cos

            m['delta_x_{}'.format(i)] = delta_x
            m['delta_y_{}'.format(i)] = delta_y

            delta_x += self.states[-i]['delta_x']
            delta_y += self.states[-i]['delta_y']
            cum_centiseconds += self.states[-i]['centiseconds']

        return m

    def col_map(self, val, threshold):
        """
        Which column of the `val_scale` matrix (see the `get_features`
        function) we are in, based on the value of the feature and the
        threshold. This lookup is the `g()` feature/action convolution
        mentioned in the paper.

        Parameters
        ----------
        val : float
            A feature value
        threshold : float
            A threshold on the feature, used to define whether we are to the
            left or right of a target.

        Returns
        -------
        int
            which column of the `val_scale` matrix we are in

        Raises
        ------
        ValueError
            Safety check against bad programming. Cannot happen.
        """
        if val < -threshold:
            return 0
        elif -threshold <= val <= threshold:
            return 1
        elif val > threshold:
            return 2
        else:
            raise ValueError('val {}, threshold {}'.format(val, threshold))

    def get_features(self, meta, action):
        """
        Given a state and an action, return
        `g(feature, threshold, f(action, speed_x,y))` from the paper.
        Essentially, make the feature itself non-linear so that it can be used
        by a linear regression.

        Parameters
        ----------
        meta : dict
            The state that we measured from the image
        action : int
            noop, left or right

        Returns
        -------
        numpy array of float
            The features that describe the state and action

        Raises
        ------
        ValueError
            Protection against sloppy programming. Cannot happen.
        """
        # value scaling (this is the `g()` function from the paper)
        val_scale = {
            'right': [-2, -2, 2],
            'left': [2, -2, -2],
            'noop_ld': [-1, 0, 1],
            'noop_d': [0, 2, 0],
            'noop_rd': [1, 0, -1],
            'noop_lr': [-3, -3, -3],
            'noop_none': [0, 2, 0],
        }

        # transform the action with the `f()` function form the paper
        # get the action
        left = action == agents.base.POSSIBLE_ACTIONS['left']
        noop = action == agents.base.POSSIBLE_ACTIONS['noop']
        right = action == agents.base.POSSIBLE_ACTIONS['right']

        # define noop type
        speed_x = meta.get('speed_x_2', 0.0)
        speed_y = meta.get('speed_y_2', 0.0)

        noop_ld = noop and speed_x < 0 and speed_y > 0
        noop_d = noop and speed_x == 0 and speed_y > 0
        noop_rd = noop and speed_x > 0 and speed_y > 0
        noop_lr = noop and np.abs(speed_x) > 0 and speed_y == 0
        noop_none = noop and speed_x == 0 and speed_y == 0

        if left:
            state = 'left'
        elif right:
            state = 'right'
        elif noop_ld:
            state = 'noop_ld'
        elif noop_d:
            state = 'noop_d'
        elif noop_rd:
            state = 'noop_rd'
        elif noop_lr:
            state = 'noop_lr'
        elif noop_none:
            state = 'noop_none'
        else:
            raise ValueError('Action: {}, Speed X: {}, Speed Y: {}'.format(
                action, speed_x, speed_y
            ))

        # now, calculate features with `g()` and `f()`
        f = []

        skier = meta.get('skier', np.array([0, 0]))
        flag = meta.get('flag', np.array([0, 0]))
        # x distance to flag
        val = (skier[0] - flag[0])

        half_slalom = features.POLE_TO_POLE / 2.0  # 16
        threshold = half_slalom

        col = self.col_map(val, threshold)
        scaled = val
        f_val = val_scale[state][col] * (np.abs(scaled) + 0.0001)
        f.append(f_val)

        # x distance from mid (edge)
        mid = features.WINDOW_WIDTH / 2.0
        val = (skier[0] - mid)
        threshold = 60
        col = self.col_map(val, threshold)
        scaled = val  # / mid
        f_val = val_scale[state][col] * (np.abs(scaled) + 0.0001)
        f.append(f_val)

        # encourage the skier to keep pointing itself downhill
        val = meta.get('cos_movement_2', 0.0)
        threshold = 0.44
        col = self.col_map(val, threshold)
        scaled = val  # / mid
        f_val = val_scale[state][col] * (np.abs(scaled) + 0.0001)
        f.append(f_val)

        return np.array(f, dtype=np.float).reshape(1, -1)

    def getQ(self, sa_features):
        """
        Get the predicted q-value for a state, action pair

        Parameters
        ----------
        sa_features : numpy array of float
            The features to predict Q with

        Returns
        -------
        float
            prediction of the Q-value for this state and action
        """
        X = self.scaler.transform(sa_features)
        return self.ridge.predict(X).ravel()[0]

    @staticmethod
    def softmax(x):
        """
        Take the sigmoid of an array of numbers

        Parameters
        ----------
        x : numpy array

        Returns
        -------
        numpy array
        """
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = np.exp(x - x.max(axis=1, keepdims=True))
        return x / x.sum(axis=1, keepdims=True)

    def sample_action(self, qs):
        """
        Given a list of Q values, randomly sample one of the actions,
        weighted by the q values
        """
        # take the softmax
        # but use the inverse prob to get the softmin
        soft_qs = (1.0 - self.softmax(qs).ravel()) / 2.0

        try:
            return np.random.choice(self.actions(), p=soft_qs)
        except:
            raise ValueError('qs {}, soft qs {}'.format(qs, soft_qs))

    def act(self, state, centiseconds, *args, **kwargs):
        """
        Choose an action based on the weights learned by function approximation

        This algorithm will produce an action given a state.
        Here we use the epsilon-greedy algorithm: with probability
        explore_prob, take a random action.

        Parameters
        ----------
        state : numpy array (250, 160)
            An green-channel image of the game board

        centiseconds : int
            How many centiseconds of game time have passed since the last state

        Returns
        -------
        int
            An action from self.actions
        """
        self.states.append(self.make_metadata(state, centiseconds))

        if log.getEffectiveLevel() <= logging.DEBUG:
            self.log_weights()

        if self.learning:
            # set exploration prob to 0 to use alternate methods
            explore_prob = (
                1.0 / (1.0 + np.sqrt(max(0, self.n_fake_episodes - 50)))
            )
            if np.random.random() < explore_prob:
                action = np.random.randint(self.n_actions)
                # log.debug('-' * 50)
            else:
                action = self.sample_action(np.array([
                    self.getQ(self.get_features(self.states[-1], a))
                    for a in self.actions()
                ]))

            # save this for the experience replay when the fake episode ends
            self.chosen_actions.append(action)
            self.real_episode_length += 1
            self.fake_episode_length += 1
        else:
            action = np.argmin(
                self.getQ(self.get_features(self.states[-1], a))
                for a in self.actions()
            )

        return action

    def react(self,
              state,
              action,
              reward,
              done,
              new_state,
              centiseconds,
              *args,
              **kwargs):
        """
        Learn from past (s, a, r, s') tuples if we have collected enough steps
        to trigger another round of fitted q-iteration.

        Parameters
        ----------
        state : numpy array (250, 160)
            A green-channel image of the game board
        int
            An action from self.actions
        reward : float
            Will be a negative value, either between -3 and -7 while going
            down the hill, or between 0 and -39999 when at the end (based
            on how many slaloms you went through).
        done : bool
            Whether this episode is done (you made it to the bottom)
        new_state : numpy array (250, 160)
            A green-channel image of the game board
        centiseconds : int
            How many centiseconds of game time have passed since the last state
        """
        # get metadata for the new state
        new_state_metadata = self.make_metadata(new_state, centiseconds)

        # be able to set the agent to use the learned policy
        if not self.learning:
            return

        # if new fake episode,
        # or it's been too long since a new fake episode
        if done or self.fake_episode_length == self.iteration_size:

            # get the fake episode's sars info
            states = (
                self.states[-self.fake_episode_length:] + [new_state_metadata]
            )

            # costs come from the transition to a new state
            # so we drop the initial state
            new_costs = self.make_fake_costs(states[1:])

            actions = self.chosen_actions

            self.sars.extend(zip(states[:-1], actions, new_costs, states[1:]))

            # keep enough around for history
            self.states = self.states[-50:]
            self.chosen_actions = []

            self.n_trainings += 1

            # create the NFQ training patterns
            # log.debug('Creating inputs')
            inputs = [
                self.get_features(sars[0], sars[1]) for sars in self.sars
            ]

            # log.debug('Creating targets')
            targets = self.create_targets(self.sars)

            # TODO: think about how we might add fake inputs/targets

            # train Q on the patterns
            # log.debug('Updating Q')
            self.update_Q(inputs, targets)

            # reset the fake episode
            self.fake_episode_length = 0
            self.n_fake_episodes += 1

        if done:
            self.episode_hits = 0
            self.real_episode_length = 0
            self.states = []

    def make_fake_costs(self, states):
        """
        Pseudo rewards

        Parameters
        ----------
        states : iterable of dict
             A list of previously measured states

        Returns
        -------
        numpy array
            The rewards for each state -> state transition
        """
        zeros = np.zeros(len(states), dtype=np.float)

        # punish every step by the average inverse speed
        time = sum(meta['centiseconds'] for meta in states)
        distance = sum(meta.get('delta_y', 0.0) for meta in states)
        sloth = zeros + (time / (distance + 0.0001))
        # cost for passing slaloms
        slalom = np.array([
            -500.0 * meta.get('score_changed', 0) for meta in states
        ])

        # # punish for flag distance
        # off_track = np.array(
        #     [meta['x_distance_to_flag'] / 5.0 for meta in states]
        # )

        cost = np.sum([
            zeros,
            sloth,
            slalom,
            # off_track
        ], axis=0)

        return cost

    def create_targets(self, sars):
        """
        sars is an iterable of 4 tuples
        Create the target values. For every s, a, r, s':
            target =  r + discount * min_{a' \in Actions} Q(s', a')
        """
        rewards = [_sars[2] for _sars in sars]
        s_prime = [_sars[3] for _sars in sars]

        # collect features for every s' with every possible action
        feats = []
        for meta in s_prime:
            # get features for every possible action in every possible state
            feats.extend([
                self.get_features(meta, action)
                for action in self.actions()
            ])

        # get Q predictions for all s' and actions
        # qs = self.net.call(np.vstack(sa_prime))
        self.scaler = sklearn.preprocessing.StandardScaler()
        X = self.scaler.fit_transform(np.vstack(feats))
        qs = self.ridge.predict(X)

        # reshape so that Qs for each s' is on a row
        # with Q for each (s', action) in the columns
        # then take the min of each row and ravel
        qs = qs.reshape(-1, self.n_actions).min(axis=1).ravel()

        # targets = reward + discount * Qopt
        targets = (qs * self.discount) + rewards

        return targets

    def update_Q(self, inputs, targets):
        """
        Train the Q approximator with the batch of inputs and targets from
        the current Q approximator
        """
        X = np.vstack(inputs)
        y = np.asarray(targets)

        # shuffle
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]

        # scale the features
        X = self.scaler.transform(X)

        self.ridge.partial_fit(X, y)

    def log_weights(self):
        """
        This mess of code was how I was watching the features change as the
        simulation ran. Please ignore it.
        """
        return
        names = np.array(['flag', 'edge', 'cos'])
        f = [
            np.argsort(np.multiply(
                self.scaler.transform(
                    self.get_features(
                        self.states[-1], self.possible_actions[action]
                    )
                ).ravel(),
                self.ridge.coef_
            ))[::-1]
            for action in ['right', 'noop', 'left']
        ]

        np.set_printoptions(precision=3, linewidth=160, suppress=True)
        right = [
            '{0} {1:.3f} {2:.3f} {3:.3f}'.format(n, fv, sfv, c)
            for n, fv, sfv, c in zip(
                names[f[0]],
                self.get_features(
                    self.states[-1], self.possible_actions['right']
                ).ravel()[f[0]],
                self.ridge.coef_[f[0]],
                np.multiply(self.scaler.transform(self.get_features(
                    self.states[-1], self.possible_actions['right']
                )).ravel(), self.ridge.coef_)[f[0]]
            )]
        noop = [
            '{0} {1:.3f} {2:.3f} {3:.3f}'.format(n, fv, sfv, c)
            for n, fv, sfv, c in zip(
                names[f[1]],
                self.get_features(
                    self.states[-1], self.possible_actions['noop']
                ).ravel()[f[1]],
                self.ridge.coef_[f[1]],
                np.multiply(self.scaler.transform(self.get_features(
                    self.states[-1], self.possible_actions['noop']
                )).ravel(), self.ridge.coef_)[f[1]]
            )]
        left = [
            '{0} {1:.3f} {2:.3f} {3:.3f}'.format(n, fv, sfv, c)
            for n, fv, sfv, c in zip(
                names[f[2]],
                self.get_features(
                    self.states[-1], self.possible_actions['left']
                ).ravel()[f[2]],
                self.ridge.coef_[f[2]],
                np.multiply(self.scaler.transform(self.get_features(
                    self.states[-1], self.possible_actions['left']
                )).ravel(), self.ridge.coef_)[f[2]]
            )]

        winner = np.argmin([
            self.getQ(self.get_features(self.states[-1], a))
            for a in self.actions()
        ])

        log.debug('\n<--: {} {}\nnoop: {} {}\n-->: {} {}'.format(
            right,
            '+' if winner == self.possible_actions['right'] else '',
            noop,
            '+' if winner == self.possible_actions['noop'] else '',
            left,
            '+' if winner == self.possible_actions['left'] else ''
        ))

        np.set_printoptions()

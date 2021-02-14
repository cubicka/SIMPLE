import numpy as np

from gym import Env, spaces

from .sushi.game import SushiGo

class SushiEnv(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, verbose = False):
        super(SushiEnv, self).__init__()
        self.game = SushiGo()
        self.name = 'sushi'
        self.n_players = 5

        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(0, np.inf, (171 + 17 ,))
        self.verbose = verbose

    @property
    def observation(self):
        return np.append(self.game.observation(), self.legal_actions)

    @property
    def current_player_num(self):
        return self.game.current_player_num

    @property
    def legal_actions(self):
        legalActions = np.zeros(self.action_space.n)
        for action in self.game.legalActions():
            legalActions[action] = 1
        return legalActions

    def reset(self, nPlayers=None):
        self.game.reset(nPlayers)
        return self.observation

    def step(self, action):
        if self.legal_actions[action] == 0:
            reward = np.array([1.0/(self.n_players-1)] * self.n_players)
            reward[self.current_player_num] = -1.0

            # selectedAction = np.random.choice(self.game.legalActions())
            # self.game.step(selectedAction)
            return self.observation, reward, True, {}

        self.game.step(action)
        return self.observation, self.game.returns(), self.game.isTerminal(), {}

    def render(self, mode='human', close=False):
        if close or not self.verbose:
        # if close:
            return
        self.game.render()

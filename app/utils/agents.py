import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import random
import string

import config

from stable_baselines import logger

def sample_action(action_probs):
    action = np.random.choice(len(action_probs), p = action_probs)
    return action


def mask_actions(legal_actions, action_probs):
    masked_action_probs = np.multiply(legal_actions, action_probs)
    actionProbSum = np.sum(masked_action_probs)
    if actionProbSum > 0:
      masked_action_probs = masked_action_probs / np.sum(masked_action_probs)
    else:
      masked_action_probs = legal_actions / np.sum(legal_actions)
    return masked_action_probs





class Agent():
  def __init__(self, name, model = None):
      self.name = name
      self.id = self.name + '_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
      self.model = model
      self.points = 0

  def print_top_actions(self, action_probs, printer=None):
    if printer is None:
      printer = logger.debug
    top5_action_idx = np.argsort(-action_probs)[:5]
    top5_actions = action_probs[top5_action_idx]
    printer(f"Top 5 actions: {[str(i) + ': ' + str(round(a,2))[:5] for i,a in zip(top5_action_idx, top5_actions)]}")

  def choose_action(self, env, choose_best_action, mask_invalid_actions, printer=None):
      if printer is None:
        printer = logger.debug
      if self.name == 'rules':
        action_probs = np.array(env.rules_move())
        value = None
      else:
        action_probs = self.model.action_probability(env.observation)
        value = self.model.policy_pi.value(np.array([env.observation]))[0]
        printer(f'Value {value:.2f}')

      # self.print_top_actions(action_probs)
      
      if mask_invalid_actions:
        action_probs = mask_actions(env.legal_actions, action_probs)
        # logger.debug('Masked ->')
        self.print_top_actions(action_probs, printer)

      if not choose_best_action:
          action = sample_action(action_probs)
          printer(f'Sampled action {action} chosen')
      else:
          action = np.argmax(action_probs)
          printer(f'Best action {action}')

      return action

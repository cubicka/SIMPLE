import numpy as np

from numba import jit

from .card import Card, cardToId

@jit(nopython=True)
def removeCard(cards, card):
    for idx, c in enumerate(cards):
        if c == card:
            return np.delete(cards, [idx])
    return cards

@jit(nopython=True)
def countCard(cards, card):
    return len(cards[cards == card])

@jit(nopython=True)
def countMaki(cards):
    maki1 = countCard(cards, cardToId(Card.MakiRoll))
    maki2 = countCard(cards, cardToId(Card.MakiRolls2))
    maki3 = countCard(cards, cardToId(Card.MakiRolls3))
    return maki1 + 2*maki2  + 3*maki3

@jit(nopython=True)
def hasCard(cards, card):
    return countCard(cards, card) > 0

@jit(nopython=True)
def addCard(cards, card):
    return np.append(cards, [card])
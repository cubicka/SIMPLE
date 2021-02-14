import numpy as np

from numba import types
from numba.experimental import jitclass

from .card import Card, cardToId, canBePutIntoWasabi, putIntoWasabi
from .util import addCard, countCard, hasCard, removeCard

spec = [
    ('hands', types.int64[:]),
    ('played', types.int64[:]),
    ('pick', types.int64[:]),
    ('score', types.int64),
]

@jitclass(spec)
class Player():
    def __init__(self):
        self.hands = np.zeros(0, dtype=types.int64)
        self.played = np.zeros(0, dtype=types.int64)
        self.pick = np.zeros(0, dtype=types.int64)
        self.score = 0

    def getCards(self, cards):
        self.hands = np.sort(cards)

    def pickCard(self, card):
        self.pick = np.append(self.pick, np.array(card))

    def playIndividualCard(self, cardId):
        self.hands = removeCard(self.hands, cardId)
        if canBePutIntoWasabi(cardId) and hasCard(self.played, cardToId(Card.Wasabi)):
            withWasabi = putIntoWasabi(cardId)
            self.played = removeCard(self.played, cardToId(Card.Wasabi))
            self.played = addCard(self.played, withWasabi)
            return
        
        self.played = addCard(self.played, cardId)
        return

    def playCard(self):
        for pickedCard in self.pick:
            self.playIndividualCard(pickedCard)

        if len(self.pick) > 1:
            self.played = removeCard(self.played, cardToId(Card.Chopsticks))
            self.hands = addCard(self.hands, cardToId(Card.Chopsticks))

        self.pick = np.zeros(0, dtype=types.int64)

    def endRoundScoring(self):
        tempuraCount = countCard(self.played, cardToId(Card.Tempura))
        self.score += (tempuraCount // 2) * 5

        sashimiCount = countCard(self.played, cardToId(Card.Sashimi))
        self.score += (sashimiCount // 3) * 10

        dumplingCount = countCard(self.played, cardToId(Card.Dumpling))
        if dumplingCount == 1: self.score += 1
        elif dumplingCount == 2: self.score += 3
        elif dumplingCount == 3: self.score += 6
        elif dumplingCount == 4: self.score += 10
        elif dumplingCount == 5: self.score += 15

        squidNigiriWasabiCount = countCard(self.played, cardToId(Card.SquidNigiriWasabi))
        self.score += squidNigiriWasabiCount * 9

        salmonNigiriWasabiCount = countCard(self.played, cardToId(Card.SalmonNigiriWasabi))
        self.score += salmonNigiriWasabiCount * 6

        eggNigiriWasabiCount = countCard(self.played, cardToId(Card.EggNigiriWasabi))
        self.score += eggNigiriWasabiCount * 3

        squidNigiriCount = countCard(self.played, cardToId(Card.SquidNigiri))
        self.score += squidNigiriCount * 3

        salmonNigiriCount = countCard(self.played, cardToId(Card.SalmonNigiri))
        self.score += salmonNigiriCount * 2

        eggNigiriCount = countCard(self.played, cardToId(Card.EggNigiri))
        self.score += eggNigiriCount

    def discardCard(self):
        self.played = self.played[self.played == cardToId(Card.Pudding)]
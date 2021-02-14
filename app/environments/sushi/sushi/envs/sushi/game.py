import numpy as np

from numba import jit, typed, types
from numba.experimental import jitclass

from .card import Card, getCards, getCardCounts, cardToId, idToCard
from .observation import getObservation, renderObs
from .player import Player
from .turnStatus import TurnStatus
from .util import addCard, countCard, countMaki, hasCard, removeCard

spec = [
    ('name', types.string),
    ('n_players', types.int64),
    ('currentNPlayers', types.int64),
    ('cards', types.int64[:]),
    ('discards', types.int64[:]),
    ('players', types.ListType(Player.class_type.instance_type)),
    ('current_player_num', types.int64),
    ('turnTaken', types.int64),
    ('round', types.int64),
    ('turnStatus', types.int64),
    ('actionLen', types.int64),
]

@jitclass(spec)
class SushiGo():
    def __init__(self):
        self.name = 'sushiX'
        self.n_players = 5

    def reset(self, nPlayers=None):
        if nPlayers is None:
            self.currentNPlayers = np.random.randint(2, self.n_players + 1)
        else:
            self.currentNPlayers = nPlayers
        self.players = typed.List([Player() for _ in range(self.currentNPlayers)])
        self.current_player_num = 0
        self.turnTaken = 0
        self.round = 0
        self.turnStatus = TurnStatus.FirstPick.value
        self.actionLen = len(getCards())

        setup(self)

    def observation(self):
        return getObservation(self)

    def legalActions(self):
        if self.isTerminal(): return np.zeros(0, dtype=types.int64)
        if self.turnStatus == TurnStatus.AskForChopstick.value:
            return np.array([self.actionLen, self.actionLen + 1])

        currentPlayer = self.players[self.current_player_num]
        if self.turnStatus == TurnStatus.FirstPick.value:
            # return np.unique(currentPlayer.hands)
            return currentPlayer.hands

        # return np.unique(removeCard(currentPlayer.hands, currentPlayer.pickCard[0]))
        return removeCard(currentPlayer.hands, currentPlayer.pick[0])

    def step(self, action):
        # notAChoice = True
        # for a in self.legalActions():
        #     if a == action:
        #         notAChoice = False
        #         break

        # if notAChoice:
        #     raise Exception("Wrong action --__--")

        if self.turnStatus == TurnStatus.AskForChopstick.value:
            if action == self.actionLen:
                self.continueToNextPlayer()
                return

            self.turnStatus = TurnStatus.ChopstickPick.value
            return

        currentPlayer = self.players[self.current_player_num]
        currentPlayer.pickCard(action)

        hasChopstick = len(currentPlayer.played[currentPlayer.played == cardToId(Card.Chopsticks)]) > 0
        if self.turnStatus != TurnStatus.FirstPick.value or not hasChopstick:
            self.continueToNextPlayer()
            return

        self.turnStatus = TurnStatus.AskForChopstick.value
        return

    def isTerminal(self):
        return self.round == 3

    def returns(self):
        if not self.isTerminal(): return np.array([0.0] * 5)
        sortedPlayers = sorted(self.players[:self.currentNPlayers], key=playerRank)

        if playerRank(sortedPlayers[0]) == playerRank(sortedPlayers[-1]): return np.array([0.0] * 5)

        bestPlayerCount = 0
        for id in range(self.currentNPlayers):
            if playerRank(sortedPlayers[id]) != playerRank(sortedPlayers[0]): break
            bestPlayerCount += 1

        worstPlayerCount = 0
        for id in range(self.currentNPlayers-1, -1, -1):
            if playerRank(sortedPlayers[id]) != playerRank(sortedPlayers[-1]): break
            worstPlayerCount += 1

        scores = [0.0] * 5
        for id in range(self.currentNPlayers):
            if playerRank(self.players[id]) == playerRank(sortedPlayers[0]):
                scores[id] = 1.0 / bestPlayerCount
            if playerRank(self.players[id]) == playerRank(sortedPlayers[-1]):
                scores[id] = -1.0 / worstPlayerCount

        return np.array(scores)

    def render(self):
        print('======= Current Player', self.current_player_num)
        renderObs(self.observation())
        print('======= Scores:')
        for player in self.players:
            print(player.score)
        # for player in self.players:
        #     if len(player.hands[player.hands != cardToId(Card.Pudding)]) > 10 or len(player.played[player.played != cardToId(Card.Pudding)]) > 10:
        #         raise Exception("Here!")

    def continueToNextPlayer(self):
        self.current_player_num += 1
        self.turnStatus = TurnStatus.FirstPick.value

        if self.current_player_num == self.currentNPlayers:
            for player in self.players:
                player.playCard()

            self.current_player_num = 0
            self.turnTaken += 1
            self.moveHands()

            if len(self.players[0].hands) == 1:
                for player in self.players:
                    player.pickCard(player.hands[0])
                    player.playCard()

                self.endRound()
                if self.round == 3:
                    self.endGame()
                else:
                    dealCards(self)


    def moveHands(self):
        endHands = self.players[self.currentNPlayers-1].hands.copy()
        for id in range(self.currentNPlayers-1, -1, -1):
            self.players[id].hands = self.players[id-1].hands
        self.players[0].hands = endHands

    def endRound(self):
        self.round += 1
        self.turnTaken = 0
        self.makiScoring()

        populateDiscard(self)
        for player in self.players:
            player.endRoundScoring()
            player.discardCard()

    def makiScoring(self):
        makiCounts = [countMaki(player.played) for player in self.players]
        bestCount, nBestCount = 0, 0
        for count in makiCounts:
            if count > bestCount:
                bestCount, nBestCount = count, 1
            elif count == bestCount:
                nBestCount += 1

        # print(bestCount, nBestCount, 6//bestCount)
        if bestCount == 0: return
        for idx, count in enumerate(makiCounts):
            if count == bestCount: self.players[idx].score += (6//nBestCount)

        if nBestCount > 1: return

        secondBestCount, nBestCount = 0, 0
        for count in makiCounts:
            if count > secondBestCount and count < bestCount:
                secondBestCount, nBestCount = count, 1
            elif count == secondBestCount:
                nBestCount += 1

        if secondBestCount == 0: return
        for idx, count in enumerate(makiCounts):
            if count == secondBestCount: self.players[idx].score += (3//nBestCount)

    def endGame(self):
        puddingCounts = [countCard(player.played, cardToId(Card.Pudding)) for player in self.players]
        bestPudding, worstPudding = 0, 1000
        for count in puddingCounts:
            if bestPudding < count: bestPudding = count
            if worstPudding > count: worstPudding = count

        if bestPudding == worstPudding: return

        bestCount = 0
        for count in puddingCounts:
            if bestPudding == count: bestCount += 1
        for idx, count in enumerate(puddingCounts):
            if count == bestPudding: self.players[idx].score += (6//bestCount)

        if self.currentNPlayers == 2: return

        worstCount = 0
        for count in puddingCounts:
            if worstPudding == count: worstCount += 1
        for idx, count in enumerate(puddingCounts):
            if count == worstPudding: self.players[idx].score -= (6//worstCount)

@jit(nopython=True)
def setup(sushi):
    setupCards(sushi)
    dealCards(sushi)

@jit(nopython=True)
def setupCards(sushi):
    cardCounts = getCardCounts()
    for idx, count in enumerate(cardCounts):
        if idx == 0:
            cards = np.array([cardToId(count[0])] * count[1])
        else:
            cards = np.append(cards, np.array([cardToId(count[0])] * count[1]))
    
    np.random.shuffle(cards)
    sushi.cards = cards
    sushi.discards = np.zeros(0, dtype=types.int64)

@jit(nopython=True)
def dealCards(sushi):
    dealCountMap = ((2, 10), (3,9), (4,8), (5,7))
    for countMap in dealCountMap:
        if countMap[0] == sushi.currentNPlayers:
            dealCount = countMap[1]

    for player in sushi.players:
        player.getCards(sushi.cards[:dealCount])
        sushi.cards = np.delete(sushi.cards, np.arange(dealCount))

@jit(nopython=True)
def playerRank(player):
    return (-player.score, -countCard(player.played, cardToId(Card.Pudding)))

@jit(nopython=True)
def populateDiscard(sushi):
    for player in sushi.players:
        sushi.discards = np.append(sushi.discards, player.played[player.played != cardToId(Card.Pudding)])

    while hasCard(sushi.discards, cardToId(Card.SalmonNigiriWasabi)):
        sushi.discards = removeCard(sushi.discards, cardToId(Card.SalmonNigiriWasabi))
        sushi.discards = addCard(sushi.discards, cardToId(Card.SalmonNigiri))
        sushi.discards = addCard(sushi.discards, cardToId(Card.Wasabi))

    while hasCard(sushi.discards, cardToId(Card.SquidNigiriWasabi)):
        sushi.discards = removeCard(sushi.discards, cardToId(Card.SquidNigiriWasabi))
        sushi.discards = addCard(sushi.discards, cardToId(Card.SquidNigiri))
        sushi.discards = addCard(sushi.discards, cardToId(Card.Wasabi))

    while hasCard(sushi.discards, cardToId(Card.EggNigiriWasabi)):
        sushi.discards = removeCard(sushi.discards, cardToId(Card.EggNigiriWasabi))
        sushi.discards = addCard(sushi.discards, cardToId(Card.EggNigiri))
        sushi.discards = addCard(sushi.discards, cardToId(Card.Wasabi))

@jit(nopython=True)
def test():
    sushi = SushiGo()
    sushi.reset()

    # print("Cards:")
    # print(sushi.cards)
    # # print([idToCard(card) for card in sushi.cards])

    # totalCount = 0
    # print("Player Hands:")
    # for player in sushi.players:
    #     totalCount += len(player.hands)
    #     print(player.hands)

    # totalCount += len(sushi.cards)
    # assert totalCount == 108, 'Card count is incorrect'

    # print("Observation: ")
    # print(getObservation(sushi))
    # sushi.turnTaken += 1
    # print(getObservation(sushi))
    # sushi.turnTaken += 1
    # print(getObservation(sushi))

    # renderObs(getObservation(sushi))

    # print("Maki counting:")
    # sushi.reset()
    # sushi.currentNPlayers = 5
    # sushi.players[0].played = np.array([0, 0, 1, 4, 6, 6])
    # sushi.players[0].endRoundScoring()
    # sushi.players[1].played = np.array([2, 4, 4, 5, 6, 8, 14])
    # sushi.players[1].endRoundScoring()
    # sushi.players[2].played = np.array([1, 2, 2, 3, 4, 7])
    # sushi.players[2].endRoundScoring()
    # sushi.players[3].played = np.array([1, 3, 3, 5, 10, 10])
    # sushi.players[3].endRoundScoring()
    # sushi.players[4].played = np.array([0, 1, 1, 3, 6, 8])
    # sushi.players[4].endRoundScoring()
    # sushi.makiScoring()

    print("Chopstick Simulation")
    sushi.reset()
    sushi.turnTaken = 5
    sushi.currentNPlayers = 5
    sushi.players[0].hands = np.array([0, 1])
    sushi.players[0].played = np.array([0, 4, 6, 12, 14])
    sushi.players[1].hands = np.array([4, 8])
    sushi.players[1].played = np.array([2, 4, 5, 5, 6])
    sushi.players[2].hands = np.array([2, 3])
    sushi.players[2].played = np.array([1, 2, 3, 7])
    sushi.players[3].hands = np.array([1, 6])
    sushi.players[3].played = np.array([3, 5, 10, 10, 12])
    sushi.players[4].hands = np.array([3, 6])
    sushi.players[4].played = np.array([0, 1, 1, 8, 12])

    sushi.step(1)
    sushi.step(16)
    sushi.step(0)
    sushi.step(8)
    sushi.step(2)
    sushi.step(1)
    sushi.render()

    sushi.step(3)
    sushi.render()

    for player in sushi.players:
        print(player.score)

    print("Done\n")

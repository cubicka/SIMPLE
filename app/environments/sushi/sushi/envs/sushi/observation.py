import numpy as np

from numba import jit, typed, types

from .card import getCards
from .turnStatus import TurnStatus

@jit(nopython=True)
def getObservation(sushi):
    currentPlayerId = sushi.current_player_num
    obs = np.array([sushi.currentNPlayers, sushi.round, sushi.turnTaken])
    obs = addTurnStatusToObs(obs, sushi)
    obs = np.append(obs, cardsToObs(sushi.discards, True))

    for id in range(sushi.n_players):
        playerId = (id + currentPlayerId) % sushi.currentNPlayers
        canSee = sushi.turnTaken >= id
        isExists = id < sushi.currentNPlayers
        obs = np.append(obs, playerToObs(sushi.players[playerId], canSee, isExists))

    return obs

@jit(nopython=True)
def renderObs(obsRaw):
    obs = obsRaw.copy()

    nPlayers, round, turnTaken = obs[:3]
    obs = np.delete(obs, np.arange(3))
    # print("Number of players", nPlayers)
    print("Round", round, '---', "Turn", turnTaken)

    turnStatus, obs = getTurnStatusOfObs(obs)
    print("Turn Status", turnStatus)

    discards, obs = obsToCard(obs)
    print("Discards", cardsStr(discards))

    for id in range(5):
        hands, obs = obsToCard(obs)
        played, obs = obsToCard(obs)
        if id < nPlayers:
            print("Player", id+1)
            print("Hands: ", cardsStr(hands))
            print("Played: ", cardsStr(played))
    print()

    assert len(obs) == 0, "Render observation is incomplete"

@jit(nopython=True)
def cardsStr(cards):
    return ", ".join([card.value for card in cards])

@jit(nopython=True)
def cardsToObs(cards, canSee=True):
    cardList = getCards()
    obs = np.zeros(len(cardList), dtype=types.int64)
    if canSee:
        for card in cards:
            obs[card] += 1

    return obs

@jit(nopython=True)
def obsToCard(obs):
    cardList = getCards()
    nCards = len(cardList)
    cardIdxs = np.arange(nCards)

    return (
        [x for id in cardIdxs if obs[id] > 0 for x in [cardList[id]] * obs[id]],
        np.delete(obs, cardIdxs),
    )

@jit(nopython=True)
def playerToObs(player, canSee=False, isExists=True):
    return np.append(
        cardsToObs(player.hands, canSee and isExists),
        cardsToObs(player.played, isExists),
    )

@jit(nopython=True)
def addTurnStatusToObs(obs, sushi):
    turnStatusObs = [0, 0, 0]
    turnStatusObs[sushi.turnStatus] = 1
    return np.append(obs, np.array(turnStatusObs))

@jit(nopython=True)
def getTurnStatusOfObs(obs):
    if obs[0] == 1: return (TurnStatus.FirstPick, np.delete(obs, np.arange(3)))
    if obs[1] == 1: return (TurnStatus.AskForChopstick, np.delete(obs, np.arange(3)))
    return (TurnStatus.ChopstickPick, np.delete(obs, np.arange(3)))

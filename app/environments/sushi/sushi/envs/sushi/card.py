import numpy as np
from enum import Enum
from numba import jit

class Card(Enum):
    Tempura = 'Tempura'
    Sashimi = 'Sashimi'
    Dumpling = 'Dumpling'
    MakiRolls2 = '2 Maki rolls'
    MakiRolls3 = '3 Maki rolls'
    MakiRoll = '1 Maki roll'
    SalmonNigiri = 'Salmon Nigiri'
    SquidNigiri = 'Squid Nigiri'
    EggNigiri = 'Egg Nigiri'
    SalmonNigiriWasabi = 'Salmon Nigiri Wasabi'
    SquidNigiriWasabi = 'Squid Nigiri Wasabi'
    EggNigiriWasabi = 'Egg Nigiri Wasabi'
    Pudding = 'Pudding'
    Wasabi = 'Wasabi'
    Chopsticks = 'Chopsticks'

@jit(nopython=True)
def getCards():
    return (
        Card.Tempura,
        Card.Sashimi,
        Card.Dumpling,
        Card.MakiRolls2,
        Card.MakiRolls3,
        Card.MakiRoll,
        Card.SalmonNigiri,
        Card.SalmonNigiriWasabi,
        Card.SquidNigiri,
        Card.SquidNigiriWasabi,
        Card.EggNigiri,
        Card.EggNigiriWasabi,
        Card.Pudding,
        Card.Wasabi,
        Card.Chopsticks,
    )

@jit(nopython=True)
def getCardCounts():
    return (
        (Card.Tempura, 14),
        (Card.Sashimi, 14),
        (Card.Dumpling, 14),
        (Card.MakiRolls2, 12),
        (Card.MakiRolls3, 8),
        (Card.MakiRoll, 6),
        (Card.SalmonNigiri, 10),
        (Card.SquidNigiri, 5),
        (Card.EggNigiri, 5),
        (Card.Pudding, 10),
        (Card.Wasabi, 6),
        (Card.Chopsticks, 4),
    )

@jit(nopython=True)
def cardToId(card):
    return getCards().index(card)

@jit(nopython=True)
def idToCard(id):
    return getCards()[id]

@jit(nopython=True)
def canBePutIntoWasabi(cardId):
    validCards = [Card.SalmonNigiri, Card.EggNigiri, Card.SquidNigiri]
    for card in validCards:
        if cardToId(card) == cardId: return True
    return False

@jit(nopython=True)
def putIntoWasabi(cardId):
    card = idToCard(cardId)
    if card == Card.SalmonNigiri: return cardToId(Card.SalmonNigiriWasabi)
    if card == Card.EggNigiri: return cardToId(Card.EggNigiriWasabi)
    if card == Card.SquidNigiri: return cardToId(Card.SquidNigiriWasabi)
    return cardId

@jit(nopython=True)
def test():
    cards = getCards()
    for card in cards:
        # print(card)
        # print(idToCard(ca))
        assert card == idToCard(cardToId(card)), 'Card-to-id conversion is incorrect'

    cardCounts = getCardCounts()
    totalCount = 0
    for count in cardCounts:
        totalCount += count[1]
    assert totalCount == 108, 'Card count is incorrect'

    print(cards)
    print(cardCounts)

    assert canBePutIntoWasabi(cardToId(Card.SalmonNigiri)), "Nigiri can be put into Wasabi"
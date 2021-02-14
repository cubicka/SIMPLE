from numba import jit

from .card import Card, cardToId, getCards

@jit(nopython=True)
def getCardNicks():
    return (
        'tem',
        'sas',
        'dum',
        'mak2',
        'mak3',
        'mak1',
        'sal',
        'salw',
        'squ',
        'squw',
        'egg',
        'eggw',
        'pud',
        'was',
        'cho',
    )

@jit(nopython=True)
def nickToCard(s):
    if s in 'pass': return 15
    if s in 'pick': return 16
    nicks = getCardNicks()
    cards = getCards()
    for idx, n in enumerate(nicks):
        if n in s: return cardToId(cards[idx])
    return 0

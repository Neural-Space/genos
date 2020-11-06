"""
Authors:
 - Ayushman Dash <ayushman@neuralspace.ai>
 - Kushal Jain <kushal@neuralspace.ai>
"""


class King:
    def __init__(self, name: str, queen: str, allegiance: str):
        self.name = name
        self.queen = queen
        self.allegiance = allegiance

    def __repr__(self):
        return f"Name:{self.name}\nQueen:{self.queen}\nAllegiance:{self.allegiance}"


class House:
    def __init__(self, king: King, home: str, sigil: str):
        self.king = king
        self.home = home
        self.sigil = sigil

    def __repr__(self):
        return f"King:{self.king.name}\nHome:{self.home}\nSigil:{self.sigil}"


def multiply(a: float, b: float) -> float:
    return a * b

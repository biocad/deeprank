from enum import Enum


class ProteinSelectionType(Enum):
    CONTACT = 1
    RESIDUE = 2


class BaseProteinSelection:
    pass 


class ProteinContactSelection:
    def __init__(self, chain1, chain2):
        self.chain1 = chain1
        self.chain2 = chain2
        self.type = ProteinSelectionType.CONTACT


class SingleResidueSelection:
    def __init__(self, chain, number):
        self.chain = chain
        self.number = number
        self.type = ProteinSelectionType.RESIDUE


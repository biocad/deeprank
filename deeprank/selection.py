from enum import Enum


class ProteinSelectionType(Enum):
    CONTACT = 1
    RESIDUE = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class BaseProteinSelection:
    pass 


class ProteinContactSelection:
    def __init__(self, chain1, chain2):
        self.chain1 = chain1
        self.chain2 = chain2
        self.type = ProteinSelectionType.CONTACT

    def get_chain_number(self, chain_id):
        if self.chain1 == chain_id
            return 0

        elif self.chain2 == chain_id:
            return 1

        else:
            raise ValueError("no such chain: {}".format(chain_id))

    def get_chains(self):
        return [self.chain1, self.chain2]

class SingleResidueSelection:
    def __init__(self, chain, number):
        self.chain = chain
        self.number = number
        self.type = ProteinSelectionType.RESIDUE

    def get_chain_number(self, chain_id):
        if self.chain == chain_id:
            return 0

        else:
            raise ValueError("no such chain: {}".format(chain_id))

    def get_chains(self):
        return [self.chain]

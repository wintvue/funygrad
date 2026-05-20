from enum import Enum, auto


class Ops(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    POW = auto()
    MATMUL = auto()
    RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    EXP = auto()
    LOG = auto()
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    RESHAPE = auto()
    PERMUTE = auto()
    EXPAND = auto()
    MOVEMENT = auto()


UNARY_OPS = {Ops.NEG, Ops.RELU, Ops.SIGMOID, Ops.TANH, Ops.EXP, Ops.LOG, Ops.MOVEMENT}
BINARY_OPS = {Ops.ADD, Ops.SUB, Ops.MUL, Ops.DIV, Ops.POW, Ops.MATMUL}
REDUCE_OPS = {Ops.SUM, Ops.MEAN, Ops.MAX}
SHAPE_OPS = {Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND}

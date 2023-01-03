from typing import List

import numpy as np

# FLOPS_PER_WATT is FLOP_PER_JOULE 

from poet.chipsets import MKR1000
from poet.power_computation import DNNLayer

device = MKR1000


class QueryKeyValueMatrix(DNNLayer):
    # Fusing Query, Key, And Value into 1
    def __init__(self, SEQ_LEN, HIDDEN_DIM, I, ATTN_HEADS, input):
        super().__init__(
            out_shape=(3 * SEQ_LEN, I, ATTN_HEADS),  # [seq_lean X intermediate_vector_dim] for 12 heads
            depends_on=[input] if input is not None else [],
            param_count=3 * HIDDEN_DIM * I * ATTN_HEADS,
        )
        self.flop = 3 * SEQ_LEN * HIDDEN_DIM * I * ATTN_HEADS


class QKTMatrix(DNNLayer):
    # Fusing Masking and Dropout
    def __init__(self, SEQ_LEN, HIDDEN_DIM, I, ATTN_HEADS, input):
        super().__init__(
            out_shape=(SEQ_LEN, I, ATTN_HEADS), depends_on=[input] if input is not None else [], param_count=0 
        )
        self.flop = SEQ_LEN * HIDDEN_DIM * I * ATTN_HEADS + np.prod(self.out_shape) + np.prod(self.out_shape)  # QKT + mask + dropout


class Mask(DNNLayer):
    def __init__(self, input: DNNLayer):
        super().__init__(out_shape=input.out_shape, depends_on=[input] if input is not None else [], param_count=0)
        self.flop = np.prod(self.out_shape)


class QKTVMatrix(DNNLayer):
    # QKTV + Concat
    def __init__(self, SEQ_LEN, HIDDEN_DIM, I, ATTN_HEADS, input):
        super().__init__(out_shape=(SEQ_LEN, I * ATTN_HEADS), depends_on=[input] if input is not None else [], param_count=0)
        self.flop = SEQ_LEN * HIDDEN_DIM * I * ATTN_HEADS + SEQ_LEN * HIDDEN_DIM * I * ATTN_HEADS  # QKTVMatrix + Concat


class Concat(DNNLayer):
    def __init__(self, SEQ_LEN, HIDDEN_DIM, I, ATTN_HEADS, input):
        super().__init__(
            out_shape=(SEQ_LEN, I * ATTN_HEADS), depends_on=[input] if input is not None else [], param_count=HIDDEN_DIM * I * ATTN_HEADS
        )
        self.flop = SEQ_LEN * HIDDEN_DIM * I * ATTN_HEADS


class LinearLayerReLU(DNNLayer):
    def __init__(self, in_features: int, out_features: int, input: DNNLayer):
        super().__init__(
            self.find_outshape(in_features, out_features, input),
            [input] if input is not None else [],
            param_count=((in_features + 1) * out_features),
        )
        self.extra_repr_params["in_features"] = in_features
        self.extra_repr_params["out_features"] = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.flop = 2 * self.param_count + self.out_features + np.prod(self.out_shape)  # (Linear) + ReLU

    def find_outshape(self, in_features, out_features, input):
        assert len(input.out_shape) == 2 and input.out_shape[1] == in_features, f"{input.out_shape}, {in_features}"
        return (input.out_shape[0], out_features)

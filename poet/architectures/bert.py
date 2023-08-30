from poet.power_computation import (
    LinearLayer,
    ReLULayer,
    GradientLayer,
    InputLayer,
    DropoutLayer,
    FlattenLayer,
    SkipAddLayer,
    CrossEntropyLoss,
    get_net_costs,
)
from poet.power_computation_transformer import (
    QueryKeyValueMatrix,
    QKTMatrix,
    Mask,
    QKTVMatrix,
    Concat,
    LinearLayerReLU,
)


# Look-up ignored for now
def make_transformer(SEQ_LEN, HIDDEN_DIM, I, HEADS, layers):
    """
    Appends a transformer block to the list of layers.

    Parameters:
        SEQ_LEN (int): Length of the sequence.
        HIDDEN_DIM (int): Dimension of the hidden layers.
        I (int): Intermediate dimension used in calculations.
        HEADS (int): Number of attention heads.
        layers (list): List of layers to which the transformer block will be appended.
    """
    input_layer = layers[-1]
    layers.append(QueryKeyValueMatrix(SEQ_LEN, HIDDEN_DIM, I, HEADS, layers[-1]))  # Calculate Query
    layers.append(QKTMatrix(SEQ_LEN=SEQ_LEN, HIDDEN_DIM=I, I=SEQ_LEN, ATTN_HEADS=HEADS, input=layers[-1]))  # QK^T
    layers.append(QKTVMatrix(SEQ_LEN, SEQ_LEN, I, HEADS, layers[-1]))  # QK^TV
    layers.append(LinearLayer(I * HEADS, HIDDEN_DIM, layers[-1]))
    # Residual
    layers.append(SkipAddLayer(input_layer, layers[-1]))
    # FFNs
    layers.append(LinearLayerReLU(HIDDEN_DIM, HIDDEN_DIM * 4, layers[-1]))
    layers.append(LinearLayer(HIDDEN_DIM * 4, HIDDEN_DIM, layers[-1]))
    layers.append(SkipAddLayer(layers[-4], layers[-1]))
    return


def BERTBase(SEQ_LEN=512, HIDDEN_DIM=768, I=64, HEADS=12, NUM_TRANSFORMER_BLOCKS=12):
    """
    Constructs a BERT model with specified parameters.

    Parameters:
        SEQ_LEN (int, optional): Length of the input sequence. Default is 512.
        HIDDEN_DIM (int, optional): Dimension of the hidden layers. Default is 768.
        I (int, optional): Intermediate dimension used in calculations. Default is 64.
        HEADS (int, optional): Number of attention heads. Default is 12.
        NUM_TRANSFORMER_BLOCKS (int, optional): Number of transformer blocks. Default is 12.

    Returns:
        list: List of model layers.
    """
    layers = [InputLayer((SEQ_LEN, HIDDEN_DIM))]
    for _transformer in range(0, NUM_TRANSFORMER_BLOCKS):
        make_transformer(SEQ_LEN, HIDDEN_DIM, I, HEADS, layers)
    layers.append(CrossEntropyLoss(layers[-1], SEQ_LEN))
    bwd_layers = list(reversed(layers))
    for out_layer, in_layer in zip(bwd_layers[1:], bwd_layers[:-1]):
        layers.append(GradientLayer(out_layer, [in_layer], layers[-1]))
    return layers


if __name__ == "__main__":
    from poet.chipsets import *

    print("### network ###")
    SEQ_LEN = 512
    HIDDEN_DIM = 768
    I = 64  # Intermediate vector representation
    HEADS = 12
    NUM_TRANSFORMER_BLOCKS = 12
    CHIPSET = M4F
    net = BERTBase(SEQ_LEN, HIDDEN_DIM, I, HEADS, NUM_TRANSFORMER_BLOCKS)
    param_count = 0
    totalEnergy = 0
    for layer in net:
        param_count += layer.param_count
        totalEnergy += layer.energy(CHIPSET)
        print(layer)

    print("### Total number of parameters in network:", param_count)
    print("### Total energy (forward + backward) cost is:", totalEnergy)

    # print("### Profiles ###")
    # resource = get_net_costs(net, MKR1000)
    # for _list in resource:
    #     print(_list, ":", resource[_list])

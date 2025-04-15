ENCODING_2D = {
    "0000" : "A",
    "0010" : "B",
    "0001" : "C",
    "0011" : "D",
    "1000" : "E",
    "1010" : "F",
    "1001" : "G",
    "1011" : "H",
    "0100" : "I",
    "0110" : "J",
    "0101" : "K",
    "0111" : "L",
    "1100" : "M",
    "1110" : "N",
    "1101" : "O",
    "1111" : "P"
}

IMAGE_DIRECTORY = "microCT-multiview-100/microCT-multiview"
VOXEL_DIRECTORY = "microCT-voxel-filled/microCT-voxel"


SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
MIN_FREQ = 2
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
PAD_INDEX = 1

SPECIAL_TOKENS = [
    UNK_TOKEN,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
]

BATCH_SIZE = 2

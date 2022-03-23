from yacs.config import CfgNode as CN

_C = CN()

# Experiment name
_C.EXPERIMENT = ""
# Sub-sample training dataset to debug
_C.DEBUG = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
_C.SYSTEM.CUDA = True
_C.SYSTEM.DISTRIBUTED = False
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 32

_C.DIRS = CN()
_C.DIRS.DATA = "data/"
_C.DIRS.WEIGHTS = "weights/"
_C.DIRS.OUTPUTS = "outputs/"
_C.DIRS.LOGS = "logs/"
_C.DIRS.TRAIN_CSV_FILE = "train.csv"
_C.DIRS.VALID_CSV_FILE = "recognition_solution_v2.1.csv"
_C.DIRS.INDEX_RECOGNITION_CSV_FILE = "train_clean.csv"
_C.DIRS.INDEX_RETRIEVAL_CSV_FILE = "index.csv"
_C.DIRS.TEST_RECOGNITION_CSV_FILE = "recognition_solution_v2.1.csv"
_C.DIRS.TEST_RETRIEVAL_CSV_FILE = "retrieval_solution_v2.1.csv"

_C.DIRS.EXTERNAL_TRAIN_RETRIEVAL_CSV_FILE = (
    "external_train_retrieval.csv"  # images start with 0
)

_C.DIRS.LABEL_MAPPING_FILE = "train_landmark_to_label.json"

_C.DATA = CN()

_C.DATA.MIN_SIZE_TRAIN = [
    800,
]
_C.DATA.MAX_SIZE_TRAIN = 1333
_C.DATA.MIN_SIZE_TEST = [
    800,
]
_C.DATA.MAX_SIZE_TEST = 1333
_C.DATA.TEST_PREPROCESSING = "mosaic"

# Use class-balanced data sampler
_C.DATA.BALANCED = False
_C.DATA.NUM_SAMPLES_PER_CLASS = 10

# Use pseudo-labeled data
_C.DATA.SEMI_SUPERVISED = CN({"ENABLED": False})
_C.DATA.SEMI_SUPERVISED.PROB_THRESHOLD = 0.95

# Cross-validation
_C.DATA.KFOLD = CN({"ENABLED": False})
_C.DATA.NUM_FOLDS = 5
_C.DATA.FOLD = 0

_C.DATA.HEIGHT = 224
_C.DATA.WIDTH = 224
_C.DATA.SCALE = 0.75
_C.DATA.AUGMENT = "randaug"
_C.DATA.RANDAUG = CN()
_C.DATA.RANDAUG.SPATIAL_ONLY = True
_C.DATA.RANDAUG.N = 2
_C.DATA.RANDAUG.M = [1, 10]
_C.DATA.AUGMIX = CN()
_C.DATA.AUGMIX.ALPHA = 1.0
_C.DATA.AUGMIX.BETA = 1.0
_C.DATA.CUTMIX = False
_C.DATA.MIXUP = False
_C.DATA.RESIZEMIX = False
_C.DATA.RESIZEMIX_ALPHA = 0.1
_C.DATA.RESIZEMIX_BETA = 0.6
_C.DATA.CM_ALPHA = 0.5
_C.DATA.CUTOUT = False
# Interpolation (bilinear:1, bicubic: 2)
_C.DATA.INTERP = 1
_C.DATA.IN_CHANS = 3
_C.DATA.TEST_PREPROCESSING = "mosaic"

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 40
_C.TRAIN.NUM_CYCLES = 1
_C.TRAIN.BATCH_SIZE = 16

_C.TEST = CN({"ENABLED": False})

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 4
_C.OPT.BASE_LR = 1e-3
_C.OPT.BACKBONE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-3
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.EPS = 1e-4
# RMSpropTF options
_C.OPT.DECAY_EPOCHS = 2.4
_C.OPT.DECAY_RATE = 0.97
# StepLR scheduler
_C.OPT.MILESTONES = [10, 20]
# Learning rate scheduler
_C.OPT.SCHED = "cosine_warmup"
_C.OPT.SWA = CN({"ENABLED": False})
_C.OPT.SWA.DECAY_RATE = 0.999
_C.OPT.SWA.START = 10
_C.OPT.SWA.FREQ = 5

_C.LOSS = CN()
_C.LOSS.NAME = "ce"
_C.LOSS.FOCAL_LOSS_GAMMA = 2.0
_C.LOSS.LABEL_SMOOTHING_LAMBDA = 0.1

_C.MODEL = CN()
_C.MODEL.NAME = ""
# Number of classes
_C.MODEL.NUM_CLASSES = 5000
# Frozen layer(s)
_C.MODEL.FREEZE_AT = []

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.ARCH = "resnet50"
_C.MODEL.BACKBONE.OUTPUT_STRIDE = 32
_C.MODEL.BACKBONE.PRETRAINED = "imagenet"
# Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'.
_C.MODEL.BACKBONE.POOL_TYPE = "avg"
_C.MODEL.BACKBONE.GEM_P = 1.0
_C.MODEL.BACKBONE.DROPOUT = 0.2
_C.MODEL.BACKBONE.DROP_CONNECT = 0.2
_C.MODEL.BACKBONE.EMBEDDINGS_DIM = 512
_C.MODEL.FREEZE_BATCHNORM = CN({"ENABLED": False})
# Name of the layers whose outputs should be returned in forward.
_C.MODEL.BACKBONE.OUT_FEATURES = [2, 3, 4]
# OSME options
_C.MODEL.BACKBONE.OSME = CN()
_C.MODEL.BACKBONE.OSME.REDUCTION = 16
_C.MODEL.BACKBONE.OSME.NUM_EXCITES = 2

_C.MODEL.ANCHOR_GENERATOR = CN()
_C.MODEL.ANCHOR_GENERATOR.ANCHOR_SIZES = (
    (32),
    (64),
    (128),
    (256),
    (512),
)
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = (0.5, 1.0, 2.0)

# ---------------------------------------------------------------------------- #
# Metric learning options
# ---------------------------------------------------------------------------- #
_C.MODEL.ML_HEAD = CN()
_C.MODEL.ML_HEAD.NAME = "ArcFace"
_C.MODEL.ML_HEAD.SCALER = -1  # default=math.sqrt(in_features), else positive scaler
_C.MODEL.ML_HEAD.MARGIN = 0.1
# Number of centers
_C.MODEL.ML_HEAD.NUM_CENTERS = 1

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.EXTRA_BLOCKS = "p6p7"
_C.MODEL.FPN.OUT_CHANNELS = 256

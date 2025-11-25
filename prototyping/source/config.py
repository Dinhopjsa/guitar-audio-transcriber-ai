import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    PROJECT_ROOT: str        = str(Path(__file__).resolve().parent.parent)
    DATASETS_ROOT: str       = os.path.join(PROJECT_ROOT, "data", "online", "datasets", "kaggle")
    CHECKPOINTS_ROOT: str    = os.path.join(DATASETS_ROOT, "checkpoints")


@dataclass
class MLPConfig(Config):

    CHECKPOINTS_ROOT: str = os.path.join(Config.CHECKPOINTS_ROOT, "mlp")

    TARGET_SR: int                  = 11025

    SAVE_CHECKPOINT: bool           = True
    LOAD_CHECKPOINT: bool           = False

    N_MFCC: int                     = 20
    BATCH_SIZE: int                 = 32
    NORMALIZE_FEATURES: bool        = False     # note: don't use together with std scaler
    STANDARD_SCALER: bool           = True
    NORMALIZE_AUDIO_VOLUME: bool    = True

    HIDDEN_DIM: int                 = 128
    NUM_HIDDEN_LAYERS: int          = 2
    DROPOUT: float                  = 0.1

    LR: float                       = 1e-3
    DECAY: float                    = 1e-4

    EPOCHS: int                     = 50
    MAX_CLIP_NORM: float            = 1.0
    ES_WINDOW_LEN: int              = 5
    ES_SLOPE_LIMIT: float           = -0.0001



@dataclass
class CNNConfig(Config):

    CHECKPOINTS_ROOT: str = os.path.join(Config.CHECKPOINTS_ROOT, "cnn")

    TARGET_SR: int                  = 11025

    SAVE_CHECKPOINT: bool           = True
    LOAD_CHECKPOINT: bool           = False

    N_MELS: int                     = 128
    N_FFT: int                      = 512
    HOP_LENGTH: int                 = 256
    BATCH_SIZE: int                 = 32
    NORMALIZE_AUDIO_VOLUME: bool    = True


    BASE_CHANNELS: int              = 32
    NUM_BLOCKS: int                 = 3
    HIDDEN_DIM: int                 = 256
    DROPOUT: float                  = 0.1
    KERNEL_SIZE: int                = 3

    LR: float                       = 1e-3
    DECAY: float                    = 1e-4

    EPOCHS: int                     = 50
    MAX_CLIP_NORM: float            = 1.0
    ES_WINDOW_LEN: int              = 5
    ES_SLOPE_LIMIT: float           = -0.0001
    USE_AMP: bool                   = True


    #NORMALIZE_FEATURES      = False
    #STANDARD_SCALER         = True

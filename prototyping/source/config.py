import os, time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
#from tools.recording_slicer.recording_slicer import PROJECT_ROOT


@dataclass
class BaseConfig:
    # Root of the project (…/prototyping)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # Data roots
    DATASETS_ROOT: Path    = PROJECT_ROOT / "data" / "datasets" / "kaggle"

    # Source roots
    CHECKPOINTS_ROOT: Path = PROJECT_ROOT / "source" / "trainers" / "checkpoints"


@dataclass
class MLPConfig(BaseConfig):
    CHECKPOINTS_DIR: Path = BaseConfig.CHECKPOINTS_ROOT / "mlp"

    TARGET_SR: int               = 11025 * 1

    SAVE_CHECKPOINT: bool        = True
    LOAD_CHECKPOINT: bool        = False  # DEPRECIATING

    N_MFCC: int                  = 20
    BATCH_SIZE: int              = 32
    NORMALIZE_FEATURES: bool     = False  # don't use with std scaler
    STANDARD_SCALER: bool        = True
    NORMALIZE_AUDIO_VOLUME: bool = True

    HIDDEN_DIM: int              = 128
    NUM_HIDDEN_LAYERS: int       = 2
    DROPOUT: float               = 0.1

    LR: float                    = 1e-3
    DECAY: float                 = 1e-4

    EPOCHS: int                  = 30
    MAX_CLIP_NORM: float         = 1.0
    ES_WINDOW_LEN: int           = 5
    ES_SLOPE_LIMIT: float        = -0.0001


@dataclass
class CNNConfig(BaseConfig):
    CHECKPOINTS_DIR: Path = BaseConfig.CHECKPOINTS_ROOT / "cnn"

    TARGET_SR: int               = 11025

    SAVE_CHECKPOINT: bool        = True
    LOAD_CHECKPOINT: bool        = False  # DEPRECIATING

    N_MELS: int                  = 32 * 2
    N_FFT: int                   = 512
    HOP_LENGTH: int              = 256
    BATCH_SIZE: int              = 32
    NORMALIZE_AUDIO_VOLUME: bool = True

    BASE_CHANNELS: int           = 32
    NUM_BLOCKS: int              = 3
    HIDDEN_DIM: int              = 256
    DROPOUT: float               = 0.1
    KERNEL_SIZE: int             = 3

    LR: float                    = 1e-3
    DECAY: float                 = 1e-4

    EPOCHS: int                  = 30
    MAX_CLIP_NORM: float         = 1.0
    ES_WINDOW_LEN: int           = 4
    ES_SLOPE_LIMIT: float        = -0.0001
    USE_AMP: bool                = True


@dataclass
class AudioSlicerConfig(BaseConfig):
    AUDIO_NAME: str = "E2_Only"
    TIME_STAMP: str = str(datetime.now().strftime("%m-%d_%H-%M-%S"))

    IN_AUDIO_ROOT: Path = BaseConfig.PROJECT_ROOT / "data" / "inference" / "guitar_audio"
    IN_AUDIO_PATH: Path = IN_AUDIO_ROOT / AUDIO_NAME / f"{AUDIO_NAME}.wav"  # in Track

    OUT_CLIPS_ROOT: Path = BaseConfig.PROJECT_ROOT / "data" / "inference" / "guitar_note_clips"
    OUT_CLIPS_DIR: Path = OUT_CLIPS_ROOT / AUDIO_NAME / TIME_STAMP  # out Clip folder

    MIN_IN_DB_THRESHOLD: float = -45.0
    MIN_SLICE_RMS_DB: float = -37.5

    TARGET_SR: int = 11025
    HOP_LEN: int = (256 * 3)    # window size for detection - number of samples you “move forward” between frames when computing any time–frequency - relates to sensitivity
    MIN_SEP: float = 0.25       # secs
    CLIP_LENGTH: float = 0.75   # final clip duration (secs)


@dataclass
class TranscribeConfig(BaseConfig):
    INFERENCE_CLIPS_ROOT: Path = BaseConfig.PROJECT_ROOT / "data" / "inference" / "guitar_note_clips"
    INFERENCE_AUDIO_ROOT: Path = BaseConfig.PROJECT_ROOT / "data" / "inference" / "guitar_audio"

    TARGET_SR: int = 11025
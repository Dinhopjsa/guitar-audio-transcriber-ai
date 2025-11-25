
from config import *
from audio_preprocessing import *
from mlp_trainer import *
from cnn_trainer import *

import numpy as np
import torch






class NotePredictor:
    def __init__(self):

        self.mlp = None
        self.cnn = None


    # - MODELS
    def load_models(self, mlp_ckpt="mlp_ckpt.ckpt", cnn_ckpt="cnn_ckpt.ckpt", mlp_root="checkpoints/mlp/", cnn_root="checkpoints/cnn/"):
        """
        Load MLP and CNN model weights from their respective checkpoints.
        Note: This only loads the model parameters, not the trainer/optimizer.
        """

        # ---- Load MLP checkpoint ----
        mlp_path = os.path.join(mlp_root, mlp_ckpt)
        if not os.path.isfile(mlp_path):
            raise FileNotFoundError(f"[load_models] No MLP checkpoint found: {mlp_path}")

        mlp_ckpt_data = torch.load(mlp_path, map_location="cpu")

        self.mlp = MLP(**mlp_ckpt_data["model_meta"]["init_args"])

        if "model" not in mlp_ckpt_data:
            raise KeyError("[load_models] MLP checkpoint missing 'model' field")

        self.mlp.load_state_dict(mlp_ckpt_data["model"])
        print(f"[load_models] Loaded MLP model from {mlp_path}")

        # ---- Load CNN checkpoint ----
        cnn_path = os.path.join(cnn_root, cnn_ckpt)
        if not os.path.isfile(cnn_path):
            raise FileNotFoundError(f"[load_models] No CNN checkpoint found: {cnn_path}")

        cnn_ckpt_data = torch.load(cnn_path, map_location="cpu")
        if "model" not in cnn_ckpt_data:
            raise KeyError("[load_models] CNN checkpoint missing 'model' field")

        self.cnn.load_state_dict(cnn_ckpt_data["model"])
        print(f"[load_models] Loaded CNN model from {cnn_path}")

        print("[load_models] All models successfully loaded.")





    # - PREDICT & SCORE



    # - SLICING & LOADING










def main():
    predictor = NotePredictor()

    predictor.load_models()


if __name__ == "__main__":
    main()

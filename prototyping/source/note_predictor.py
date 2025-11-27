
from config import *
from audio_preprocessing import *
from mlp_trainer import *
from cnn_trainer import *

import numpy as np
import torch






class NotePredictor:
    def __init__(self, device=None):

        self.device = torch.device("cpu") #torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.mlp = None
        self.cnn = None

        self.reverse_map = None

        self.configs = {"mlp_config":None, "cnn_config":None}


    # - MODELS
    def load_models(
        self,
        mlp_ckpt: str = "mlp_ckpt.ckpt",
        cnn_ckpt: str = "cnn_ckpt.ckpt",
        mlp_root: str = "checkpoints/mlp/",
        cnn_root: str = "checkpoints/cnn/",
    ):
        """Loads: mlp, cnn, reverse_map from checkpoints."""

                # ---- Load MLP checkpoint ----
        mlp_path = os.path.join(mlp_root, mlp_ckpt)
        if not os.path.isfile(mlp_path):
            raise FileNotFoundError(f"[load_models] No MLP checkpoint found: {mlp_path}")

        mlp_ckpt_data = torch.load(mlp_path, map_location="cpu", weights_only=False)

        # initialize mlp model
        mlp_init_args = mlp_ckpt_data["model_init_args"]
        self.mlp = MLP(**mlp_init_args)

        if "model" not in mlp_ckpt_data:
            raise KeyError("[load_models] MLP checkpoint missing 'model' field")

        # load model state
        self.mlp.load_state_dict(mlp_ckpt_data["model"])
        self.mlp.to(self.device)
        self.mlp.eval()
        print(f"[load_models] Loaded MLP model from {mlp_path}")

        # optionally pick up reverse_map from MLP checkpoint (from cnn ckpt?)
        if self.reverse_map is None:
            rm = mlp_ckpt_data.get("reverse_map", None)
            if rm is not None:
                self.reverse_map = rm
                print("[load_models] Loaded reverse_map from MLP checkpoint.")

        # assign config
        self.configs["mlp_config"] = mlp_ckpt_data["config"]
        if self.configs["mlp_config"] is not None:
            print("[load_models] Loaded MLP config")


                # ---- Load CNN checkpoint ----
        cnn_path = os.path.join(cnn_root, cnn_ckpt)
        if not os.path.isfile(cnn_path):
            raise FileNotFoundError(f"[load_models] No MLP checkpoint found: {mlp_path}")

        cnn_ckpt_data = torch.load(cnn_path, map_location="cpu", weights_only=False)

        # initialize cnn model
        cnn_init_args = cnn_ckpt_data["model_init_args"]
        self.cnn = CNN(**cnn_init_args)

        if "model" not in cnn_ckpt_data:
            raise KeyError("[load_models] CNN checkpoint missing 'model' field")

        # load model state
        self.cnn.load_state_dict(cnn_ckpt_data["model"])
        self.cnn.to(self.device)
        self.cnn.eval()
        print(f"[load_models] Loaded CNN model from {cnn_path}")

        # assign config
        self.configs["cnn_config"] = cnn_ckpt_data["config"]
        if self.configs["cnn_config"] is not None:
            print("[load_models] Loaded CNN config")


                # ---- Final output ----
        print(f"[load_models] Device: {self.device}")
        if self.reverse_map is None:
            print("[load_models] Warning: reverse_map is not set; predictions will be class indices only.")



    # - PREDICT & SCORE (confidence and weighting)

    def predict(self, y_clips):
        # y_clips - array of note audio clips
        pass












def main():
    predictor = NotePredictor()

    predictor.load_models()


if __name__ == "__main__":
    main()

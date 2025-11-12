import os, argparse

from dataset_loader import AudioDatasetLoader, AVAILABLE_DATASETS, DATASETS_NUM_CLASSES, get_dataset_num_classes


# Config
SONG_PATH = "Samples/Gb_comp.wav"

SEED = 42  
DURATION = 0.5
DATASET = AVAILABLE_DATASETS[0]
NUM_CLASSES = get_dataset_num_classes(DATASET)




'''
def load_trained_mlp(hidden_dim=64, lr=0.0005, epochs=25000, dataset=DATASET, duration=DURATION):
    model = MLPClassifier(hidden_dim, lr, dataset, duration)
    model.train(epochs)
    return model


def load_trained_cnn(lr=0.001, wdecay=0, batch_size=64, test_size=0.2, epochs=50, seed=SEED, sr=44100, dataset=DATASET, dataset_loader_cls=AudioDatasetLoader, model_path=None, loader_path=None, duration=DURATION):

    if model_path and loader_path:
        trainer = trainer.load_model(
        "cnn_0.pkl", 
        "cnn_0.le.pkl", 
        dataset_name=dataset,
        dataset_loader_cls=dataset_loader_cls,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        sr=sr,
        duration=duration,
        n_mfcc=13,
        hop_length=512,
        lr=lr,
        weight_decay=wdecay
        )

    else:
        trainer = CNNTrainer(
        dataset_name=dataset,
        dataset_loader_cls=dataset_loader_cls,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        sr=sr,
        duration=duration,
        n_mfcc=13,
        hop_length=512,
        lr=lr,
        weight_decay=wdecay
        )

    trainer.train(epochs)
    return trainer


def load_trained_log_reg(dataset=DATASET):
    model = LogisticRegressionClassifier(dataset_name=dataset)
    model.train()
    return model
'''



class Transcriber():
    def __init__(self):
        pass
    
    





def main(): 

    pass

''' 
    loader = AudioDatasetLoader(duration=DURATION)
    _, _, le = loader.load_features() 
    segmenter = OnsetSegmenter(duration=DURATION)
    mfccs, times = segmenter.load_mfcc_segments(SONG_PATH)
    #audios, audios_times = segmenter.load_audio_segments(SONG_PATH)
'''

    
    
    


if __name__ == "__main__":
    main()
import torch
import librosa as rosa
from omegaconf import OmegaConf as OC
import os
from glob import glob
from tqdm import tqdm
import multiprocessing as mp

hparams = OC.load('hparameter.yaml')

def wav2pt(wav):
    # Load wav file as floating point time series
    y,_ = rosa.load(wav, sr = hparams.audio.sr, mono = True)
    # Trim leading and trailing silence 15db as threshold
    y,_ = rosa.effects.trim(y, 15)
    # Change file name from .wav to .pt
    pt_name = os.path.splitext(wav)[0]+'.pt'
    # Convert time series into a pytorch tensor
    pt = torch.tensor(y)
    # Save file to dir
    torch.save(pt ,pt_name)
    # Delete variables
    del y, pt 
    return

if __name__=='__main__':
    print("Starting")
    hparams = OC.load('hparameter.yaml')
    dir = hparams.data.dir
    # Get all dir of .flac files.
    print(dir)
    # TODO: change .flac to .wav?
    wavs = glob(os.path.join(dir, '*/*.wav'))
    pool = mp.Pool(processes = hparams.train.num_workers)
    with tqdm(total = len(wavs)) as pbar:
        print("Started processing")
        print("number of wavs: " + str(len(wavs)) )
        # Multithreaded. Convert .wav to .pt using wav2pt() function
        for _ in tqdm(pool.imap_unordered(wav2pt, wavs)):
            # Progress bar
            pbar.update()

            

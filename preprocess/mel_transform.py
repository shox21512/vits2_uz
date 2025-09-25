# import os
# import sys
# import glob
# import logging
# import argparse
# import traceback
# from tqdm import tqdm
# import torch
# import torch.multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor
# import torchaudio
# #added me
# torchaudio.set_audio_backend("soundfile")

# from utils.hparams import get_hparams_from_file, HParams
# from utils.mel_processing import wav_to_mel

# os.environ["OMP_NUM_THREADS"] = "1"
# log_format = "%(asctime)s %(message)s"
# logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio files")
#     parser.add_argument("-c", "--config", type=str, required=True, help="YAML file for configuration")
#     args = parser.parse_args()

#     hparams = get_hparams_from_file(args.config)
#     hparams.data_dir = args.data_dir
#     return hparams


# def process_batch(batch, sr_hps, n_fft, hop_size, win_size, n_mels, fmin, fmax):
#     wavs = []
#     for ifile in batch:
#         try:
#             wav, sr = torchaudio.load(ifile)
#             assert sr == sr_hps, f"sample rate: {sr}, expected: {sr_hps}"
#             wavs.append(wav)
#         except:
#             traceback.print_exc()
#             print("Failed to process {}".format(ifile))
#             return None

#     wav_lengths = torch.tensor([x.size(1) for x in wavs])
#     max_wav_len = wav_lengths.max()

#     wav_padded = torch.zeros(len(batch), 1, max_wav_len)
#     for i, wav in enumerate(wavs):
#         wav_padded[i, :, : wav.size(1)] = wav

#     spec = wav_to_mel(wav_padded, n_fft, n_mels, sr_hps, hop_size, win_size, fmin, fmax, center=False, norm=False)
#     spec = torch.squeeze(spec, 1)

#     for i, ifile in enumerate(batch):
#         ofile = ifile.replace(".wav", ".spec.pt")
#         spec_i = spec[i, :, : wav_lengths[i] // hop_size].clone()
#         torch.save(spec_i, ofile)

#     return batch


# def process_data(hps: HParams):
#     wav_fns = sorted(glob.glob(f"{hps.data_dir}/**/*.wav", recursive=True))
#     # wav_fns = wav_fns[:100]  # * Enable for testing
#     logging.info(f"Max: {mp.cpu_count()}; using 8 CPU cores")
#     logging.info(f"Preprocessing {len(wav_fns)} files...")

#     sr = hps.data.sample_rate
#     n_fft = hps.data.n_fft
#     hop_size = hps.data.hop_length
#     win_size = hps.data.win_length
#     n_mels = hps.data.n_mels
#     fmin = hps.data.f_min
#     fmax = hps.data.f_max

#     # Batch files to optimize disk I/O and computation
#     batch_size = 128  # Change as needed
#     audio_file_batches = [wav_fns[i : i + batch_size] for i in range(0, len(wav_fns), batch_size)]

#     # Use multiprocessing to speed up the conversion
#     with ProcessPoolExecutor(max_workers=8) as executor:
#         futures = [executor.submit(process_batch, batch, sr, n_fft, hop_size, win_size, n_mels, fmin, fmax) for batch in audio_file_batches]
#         for future in tqdm(futures):
#             if future.result() is None:
#                 logging.warning(f"Failed to process a batch.")
#                 return


# def get_size_by_ext(directory, extension):
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for f in filenames:
#             if f.endswith(extension):
#                 fp = os.path.join(dirpath, f)
#                 total_size += os.path.getsize(fp)

#     return total_size


# def human_readable_size(size):
#     """Converts size in bytes to a human-readable format."""
#     for unit in ["B", "KB", "MB", "GB", "TB"]:
#         if size < 1024:
#             return f"{size:.2f}{unit}"
#         size /= 1024
#     return f"{size:.2f}PB"  # PB is for petabyte, which will be used if the size is too large.


# if __name__ == "__main__":
#     from time import time

#     hps = parse_args()

#     start = time()
#     process_data(hps)
#     logging.info(f"Processed data in {time() - start} seconds")

#     extension = ".spec.pt"
#     size_spec = get_size_by_ext(hps.data_dir, extension)
#     logging.info(f"{extension}: \t{human_readable_size(size_spec)}")
#     extension = ".wav"
#     size_wav = get_size_by_ext(hps.data_dir, extension)
#     logging.info(f"{extension}: \t{human_readable_size(size_wav)}")
#     logging.info(f"Total: \t\t{human_readable_size(size_spec + size_wav)}")




import os
import sys
import glob
import logging
import argparse
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torchaudio
import torchaudio.functional as AF

from utils.hparams import get_hparams_from_file, HParams
from utils.mel_processing import wav_to_mel

# Barqarorlik
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("-c", "--config", type=str, required=True, help="YAML file for configuration")
    args = parser.parse_args()

    hparams = get_hparams_from_file(args.config)
    hparams.data_dir = args.data_dir
    return hparams

def process_batch(batch, sr_hps, n_fft, hop_size, win_size, n_mels, fmin, fmax):
    """
    Bir batch WAV -> Mel.
    Stereo -> mono, SR mos bo'lmasa resample, xatolar skip.
    """
    try:
        wavs, wav_lengths, bad = [], [], []

        for ifile in batch:
            try:
                wav, sr = torchaudio.load(ifile)  # [C, T]
                if sr != sr_hps:
                    wav = AF.resample(wav, sr, sr_hps)
                if wav.size(0) > 1:               # stereo -> mono
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.to(torch.float32).clamp_(-1.0, 1.0)
                wavs.append(wav)
                wav_lengths.append(wav.size(1))
            except Exception as e:
                bad.append((ifile, repr(e)))

        for f, err in bad:
            print(f"[WARN] Skipped (load/resample): {f} -> {err}")
        if not wavs:
            return []  # batch to'liq yaroqsiz bo'lsa

        wav_lengths = torch.tensor(wav_lengths, dtype=torch.long)
        max_len = int(wav_lengths.max())
        wav_padded = torch.zeros(len(wavs), 1, max_len, dtype=wavs[0].dtype)
        for i, w in enumerate(wavs):
            T = w.size(1)
            wav_padded[i, 0, :T] = w

        spec = wav_to_mel(
            wav_padded, n_fft, n_mels, sr_hps, hop_size, win_size, fmin, fmax,
            center=False, norm=False
        )  # [B,1,M,F]
        spec = spec.squeeze(1)  # [B, M, F]

        out_ok, j = [], 0
        for ifile in batch:
            if j >= len(wavs):
                break
            frames = int(wav_lengths[j] // hop_size)
            ofile = ifile.replace(".wav", ".spec.pt")
            try:
                os.makedirs(os.path.dirname(ofile), exist_ok=True)
                torch.save(spec[j, :, :frames].clone(), ofile)
                out_ok.append(ifile)
            except Exception as e:
                print(f"[WARN] Save failed: {ifile} -> {e}")
            j += 1

        return out_ok

    except Exception as e:
        print(f"[ERROR] Batch failed (example: {batch[0]} .. len={len(batch)}): {repr(e)}")
        return None

def process_data(hps: HParams):
    wav_fns = sorted(glob.glob(f"{hps.data_dir}/**/*.wav", recursive=True))
    
    # ENV bilan boshqarish
    workers = int(os.getenv("PREPROCESS_WORKERS", "1"))
    batch_size = int(os.getenv("PREPROCESS_BATCH", "8"))

    logging.info(f"Max: {mp.cpu_count()}; using {workers} CPU cores")
    logging.info(f"Preprocessing {len(wav_fns)} files...")

    sr = hps.data.sample_rate
    n_fft = hps.data.n_fft
    hop_size = hps.data.hop_length
    win_size = hps.data.win_length
    n_mels = hps.data.n_mels
    fmin = hps.data.f_min
    fmax = hps.data.f_max

    # LJSpeech odatda 22050 — mosligini xabardorlik uchun tekshiramiz (assert emas)
    if "ljs" in hps.__dict__.get("model_dir", "").lower() or "lj" in hps.data_dir.lower():
        if sr not in (22050, 44100):  # info only
            logging.info(f"[INFO] LJS-like dataset, sample_rate in config: {sr}")

    audio_file_batches = [wav_fns[i : i + batch_size] for i in range(0, len(wav_fns), batch_size)]

    # spawn konteksti bilan
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        futures = [
            executor.submit(process_batch, batch, sr, n_fft, hop_size, win_size, n_mels, fmin, fmax)
            for batch in audio_file_batches
        ]
        for future in tqdm(futures, total=len(futures)):
            res = future.result()
            if res is None:
                logging.warning("Failed to process a batch (see [ERROR] above).")
                # davom etamiz; to'xtatish kerak bo'lsa: return

def get_size_by_ext(directory, extension):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(extension):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size

def human_readable_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}PB"

if __name__ == "__main__":
    # spawn start method barqaror
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from time import time
    hps = parse_args()

    start = time()
    process_data(hps)
    logging.info(f"Processed data in {time() - start:.2f} seconds")

    ext = ".spec.pt"
    size_spec = get_size_by_ext(hps.data_dir, ext)
    logging.info(f"{ext}: \t{human_readable_size(size_spec)}")
    ext = ".wav"
    size_wav = get_size_by_ext(hps.data_dir, ext)
    logging.info(f"{ext}: \t{human_readable_size(size_wav)}")
    logging.info(f"Total: \t\t{human_readable_size(size_spec + size_wav)}")

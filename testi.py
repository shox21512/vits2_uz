# # import torch, glob
# # f = glob.glob(r"LJSpeech-1.1\wavs\LJ001-0001.spec.pt")[0]
# # spec = torch.load(f)
# # print(f, spec.shape)  # expect shape like [80, frames]

# #### mel spectogramni ochib ko'rish
# import torch
# mel = torch.load("LJSpeech-1.1\wavs\LJ001-0001.spec.pt", map_location="cpu")
# print(type(mel), mel.shape, mel.dtype)
# print(mel.min().item(), mel.max().item())
# print(mel[:, :5])  # birinchi 5 ta freym


# import matplotlib.pyplot as plt
# import librosa.display

# plt.figure(figsize=(10,4))
# plt.imshow(mel.numpy(), aspect="auto", origin="lower")
# plt.colorbar()
# plt.title("Mel spectrogram (.pt file)")
# plt.xlabel("Frames")
# plt.ylabel("Mel bins")
# plt.show()


# # check how many WAV files inside folder

import glob, os
wav_files = glob.glob(r"D:/Project/githubs/vits2/dataset_uz2/audio/*.pt", recursive=True)
print(f"Found {len(wav_files)} PT files")

# deleting .pt files
for f in wav_files:
    os.remove(f)
print(f"Deleted {len(wav_files)} PT files")



# # read every file type and count them
# import glob, os
# wav_files = glob.glob(r"D:/Project/githubs/vits2/dataset_uz/audio1/*", recursive=True)
# print(f"Found {len(wav_files)} audio files")
# file_types = {}
# for f in wav_files:
#     ext = os.path.splitext(f)[1].lower()
#     if ext not in file_types:
#         file_types[ext] = 0
#     file_types[ext] += 1
# print("File types and counts:", file_types) 


# #read .lnk file type and write that file name into txt file
# from glob import glob
# import os

# lnk_files = glob(r"D:/Project/githubs/vits2/dataset_uz/audio1/*.lnk", recursive=True)
# with open("lnk_files.txt", "w") as f:
#     for lnk in lnk_files:
#         file_name = os.path.basename(lnk)
#         f.write(file_name + "\n")
# print(f"Found {len(lnk_files)} .lnk files and wrote to lnk_files.txt")




# ### wav fayllarni CSV dagi utt_id lar bilan solishtirish
# import os
# import pandas as pd

# # CSV fayl manzili va WAV fayllar joylashgan papka manzili
# csv_file_path = "dataset_uz/uzbek_tts_output.csv"  # CSV fayl manzilini o'zgartiring
# wav_folder_path = "dataset_uz/audio"   # WAV fayllar papkasini manzilini o'zgartiring

# # CSV faylni o'qish
# df = pd.read_csv(csv_file_path, sep='|', header=None)
# csv_utt_ids = set(df[0].str.strip())  # Birinchi ustundagi utt_id larni to'plam sifatida olish

# # WAV fayllarni o'qish
# wav_files = [f for f in os.listdir(wav_folder_path) if f.endswith('.wav')]
# wav_utt_ids = set(f.replace('.wav', '') for f in wav_files)  # WAV fayl nomlaridan .wav kengaytmasini olib tashlash

# # CSV da yo'q WAV fayllarni topish
# missing_wavs = wav_utt_ids - csv_utt_ids

# # Natijalarni chop etish
# if missing_wavs:
#     print("CSV faylda qatnashmagan WAV fayllar:")
#     for wav in missing_wavs:
#         print(f"{wav}.wav")
# else:
#     print("Barcha WAV fayllar CSV faylda qatnashgan.")
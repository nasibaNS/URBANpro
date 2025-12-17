from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
import io
import soundfile as sf
import os
import streamlit as st

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']


numer_classes = len(classes)

class CheckAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(CheckAudio, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.flatten = nn.Flatten()

        self.second = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.flatten(x)
        x = self.second(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = torch.load('labels_1.pth')
index_to_label = {ind: lab for ind, lab in enumerate(labels)}
model = CheckAudio()
model.load_state_dict(torch.load("model_voice.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)
max_len = 100


def change_audio(waveform, sample_rate):
    if sample_rate != 16000:
        new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = new_sr(torch.tensor(waveform))
    spec = transform(waveform).squeeze(0)
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    if spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))
    return spec


check_audio = FastAPI()

st.title('URBAN MODEL')
st.text('Загрузите фудиофайл (.wav) для расспознавания модели')

file = st.file_uploader('Выбери аудифайл', type=['wav'])

if not file:
    st.warning('Загрузите фудиофайл')
else:
    st.audio(file)
    if st.button('Распознать'):
        try:
            data = file.read()
            if not data:
                raise HTTPException(status_code=400, detail='Файл пустой')
            wf, sr = sf.read(io.BytesIO(data), dtype='float32')
            wf = torch.tensor(wf).T

            spec = change_audio(wf, sr).to(device)
            with torch.no_grad():
                y_pred = model(spec)
                pred_ind = torch.argmax(y_pred, dim=1)
                if pred_ind.numel() == 1:
                    pred_ind = pred_ind.item()
                else:
                    pred_ind = pred_ind[0].item()
                pred_class = classes[pred_ind]
                st.success(f'Индекс:{pred_ind}, "класс": {pred_class}')
        except Exception as e:
            st.exception(f'{e}')


##Bibliotecas utilizadas
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio

#Cargar el archivo de audio en .wav
archivo_audio = 'escala.wav'
y, sr = librosa.load(archivo_audio)

# Calcular la transformada de Fourier de la señal de audio
D = librosa.stft(y)

#Extraer armónicos
harmonico, percussive = librosa.effects.hpss(y)
harmonico_stft = librosa.stft(harmonico)

# Extraer características de armónicos, transitorios y timbre(Notas de la escala)
cromagrama = librosa.feature.chroma_stft(y=y, sr=sr)
mfccs = librosa.feature.mfcc(y=y, sr=sr)
spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(D), sr=sr)

#Grafica las características
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(harmonico_stft), ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Armónicos')


##Aqui vienen las notas
plt.subplot(4, 1, 2)
librosa.display.specshow(cromagrama, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Cromagrama')

##MFCC para describir el timbre
plt.subplot(4, 1, 3)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCCs')

# Agregar el espectrograma de la señal de audio original
plt.subplot(4, 1, 4)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma')

plt.tight_layout()

plt.show()

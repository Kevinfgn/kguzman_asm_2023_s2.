import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Cargar el archivo de audio
archivo_audio = 'escala.wav'
y, sr = librosa.load(archivo_audio)

# Calcular la transformada de Fourier de la señal de audio
D = librosa.stft(y)

# Extraer armónicos
harmonic, percussive = librosa.effects.hpss(y)
harmonic_stft = librosa.stft(harmonic)

# Extraer características de armónicos, transitorios y timbre
# Ejemplo: calcular el espectro de frecuencia de corto plazo
chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
mfccs = librosa.feature.mfcc(y=y, sr=sr)
spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(D), sr=sr)

# Visualizar las características
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(harmonic_stft), ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Armónicos')

plt.subplot(3, 1, 2)
librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Cromagrama')

plt.subplot(3, 1, 3)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCCs')

plt.tight_layout()
plt.show()

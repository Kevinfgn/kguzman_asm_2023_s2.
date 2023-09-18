import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from scipy.integrate import cumtrapz

# Cargar el archivo de audio
archivo_audio = 'piano.wav'
y, sr = librosa.load(archivo_audio)

# Factor de cambio de tono (con 0.12 le bajaria el volumen a la señal)
factor_cambio_tono = 0.12

# Factor de cambio de fase
factor_cambio_fase = np.pi/2  # Cambiar la fase en 90 grados(pi radianes)

# Calcular la transformada de Fourier de la señal de audio original
D_original = librosa.stft(y)

# Cambiar el tono modificando la magnitud y manteniendo la fase original
D_tono_modificado = D_original * factor_cambio_tono

# Reconstruir la señal con el tono modificado con la incersa de la transformada de fourier
y_tono_modificado = librosa.istft(D_tono_modificado)

# Guardar la señal con el tono modificado como un nuevo archivo de audio
nombre_archivo_tono_modificado = 'señal_tono_piano_modificado.wav'
sf.write(nombre_archivo_tono_modificado, y_tono_modificado, sr)

print(f'Se ha guardado la señal con el tono modificado en {nombre_archivo_tono_modificado}')

# 2. Cambiar el tono en el dominio de la frecuencia y la fase
# Obtener la magnitud y la fase de la transformada de Fourier original
magnitud_original = np.abs(D_original)
fase_original = np.angle(D_original)

# Aplicar la integración compleja para cambiar la fase
fase_modificada = cumtrapz(fase_original, initial=0) + factor_cambio_fase

# Reconstruir la señal con el tono y la fase cambiados
D_modificado = magnitud_original * np.exp(1j * fase_modificada)
y_tono_fase_modificados = librosa.istft(D_modificado)

# Guardar la señal con el tono y la fase cambiados como un nuevo archivo de audio
nombre_archivo_tono_fase_modificados = 'señal_fase_piano_modificada.wav'
sf.write(nombre_archivo_tono_fase_modificados, y_tono_fase_modificados, sr)

#Grafica las señales en el dominio del tiempo
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(y)
plt.title('Señal de Audio Original')

plt.subplot(3, 1, 2)
plt.plot(y_tono_modificado)
plt.title('Señal de Audio con Tono Modificado')

plt.subplot(3, 1, 3)
plt.plot(y_tono_fase_modificados)
plt.title('Señal de Audio Fase Modificada')

plt.tight_layout()

plt.tight_layout()
plt.show()

print(f'Se ha guardado la señal con el tono y la fase modificados en {nombre_archivo_tono_fase_modificados}')

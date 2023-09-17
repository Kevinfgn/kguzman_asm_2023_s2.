import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Cargar el archivo de audio original
archivo_audio = 'escala.wav'
y_original, sr = librosa.load(archivo_audio)

# Cargar la señal con la fase modificada que generaste anteriormente
señal_fase_modificada = 'escala_fase_modificada.wav'
y_fase_modificada, sr_fase_modificada = librosa.load(señal_fase_modificada)

# Asegurarse de que ambas señales tengan la misma longitud
min_length = min(len(y_original), len(y_fase_modificada))
y_original = y_original[:min_length]
y_fase_modificada = y_fase_modificada[:min_length]

# Sumar las señales originales y modificadas en fase
y_combinada = y_original + y_fase_modificada

# Guardar la señal combinada como un nuevo archivo de audio
nombre_archivo_combinado = 'efecto_chorus.wav'
sf.write(nombre_archivo_combinado, y_combinada, sr)

# Crear un eje de tiempo
tiempo = librosa.times_like(y_original)

# Graficar las tres señales en un mismo gráfico
plt.figure(figsize=(12, 6))
plt.plot(tiempo, y_original, label='Señal Original', color='blue')
plt.plot(tiempo, y_fase_modificada, label='Señal Fase Modificada', color='green')
plt.plot(tiempo, y_combinada, label='Señal Combinada', color='red')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Señales Originales y Combinada')
plt.legend()
plt.grid(True)
plt.show()

print(f'Se ha guardado la señal combinada en {nombre_archivo_combinado}')

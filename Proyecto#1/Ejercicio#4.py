import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parámetros de la señal senoidal
amplitud = 1.0
frecuencia = 5.0  # Hz
duracion = 3.0  # segundos
muestras_por_segundo = 1000
num_muestras = int(duracion * muestras_por_segundo)

# Generar la señal senoidal
tiempo = np.linspace(0, duracion, num_muestras)
senal_senoidal = amplitud * np.sin(2 * np.pi * frecuencia * tiempo)

# Agregar ruido gaussiano
amplitud_ruido = 0.5
ruido = amplitud_ruido * np.random.randn(num_muestras)
senal_con_ruido = senal_senoidal + ruido

# Derivar la señal
derivada = np.gradient(senal_con_ruido, tiempo)

# Aplicar integración compleja para reconstruir la señal original
senal_reconstruida = np.cumsum(derivada) / muestras_por_segundo

# Aplicar un filtro pasa bajos para eliminar componentes de alta frecuencia
cutoff_frequency = 10.0  # Hz
b, a = butter(4, cutoff_frequency / (muestras_por_segundo / 2), 'low')
senal_filtrada = filtfilt(b, a, senal_reconstruida)

# Visualizar la señal senoidal original, la señal con ruido, la señal reconstruida y la señal filtrada
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(tiempo, senal_senoidal)
plt.title('Señal Senoidal Original')
plt.subplot(4, 1, 2)
plt.plot(tiempo, senal_con_ruido)
plt.title('Señal con Ruido Gaussiano')
plt.subplot(4, 1, 3)
plt.plot(tiempo, senal_reconstruida)
plt.title('Señal Reconstruida')
plt.subplot(4, 1, 4)
plt.plot(tiempo, senal_filtrada)
plt.title('Señal Filtrada')
plt.tight_layout()
plt.show()


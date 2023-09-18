import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

#Parámetros de la señal seno
amplitud = 1.0
frecuencia = 5.0  #Hz
duracion = 1.0  #segundos
muestras_por_segundo = 1000
num_muestras = int(duracion * muestras_por_segundo)

# Generar la señal senoidal
tiempo = np.linspace(0, duracion, num_muestras)
senal_senoidal = amplitud * np.sin(2 * np.pi * frecuencia * tiempo)

# Agregar ruido gaussiano
amplitud_ruido = 0.1 #Dependiendo del ruido se van a necesitar varias integrales o derivdas para reconstruir la señal
ruido = amplitud_ruido * np.random.randn(num_muestras)
senal_con_ruido = senal_senoidal + ruido

# Derivar la señal
derivada = np.gradient(senal_con_ruido, tiempo)

# Aplicar integración compleja para reconstruir la señal original
señal_reconstruida = np.cumsum(derivada) / muestras_por_segundo


#Deriva e integra hasta obtener una señal reconstruida
segunda_derivada = np.gradient(señal_reconstruida,tiempo)
señal_reconstruida_dos = np.cumsum(segunda_derivada)/muestras_por_segundo
tercera_derivada = np.gradient(señal_reconstruida_dos,tiempo)
señal_reconstruida_tres = np.cumsum(tercera_derivada)/muestras_por_segundo
cuarta_derivada = np.gradient(señal_reconstruida_tres,tiempo)
señal_reconstruida_4 = np.cumsum(cuarta_derivada)/muestras_por_segundo
quinta_derivada = np.gradient(señal_reconstruida_4,tiempo)
señal_reconstruida_5 = np.cumsum(quinta_derivada)/muestras_por_segundo
sexta_derivada = np.gradient(señal_reconstruida_5,tiempo)
señal_reconstruida_6 = np.cumsum(sexta_derivada)/muestras_por_segundo
septim_derivada = np.gradient(señal_reconstruida_6,tiempo)
señal_reconstruida_7 = np.cumsum(septim_derivada)/muestras_por_segundo

# Aplicar un filtro pasa bajos para eliminar componentes de alta frecuencia
cutoff_frequency = 10.0  # Hz
b, a = butter(4, cutoff_frequency / (muestras_por_segundo / 2), 'low')
señal_filtrada = filtfilt(b, a, señal_reconstruida)

# Visualizar la señal senoidal original, la señal con ruido, la señal reconstruida y la señal filtrada
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(tiempo, senal_senoidal)
plt.title('Señal Seno Original')
plt.subplot(4, 1, 2)
plt.plot(tiempo, senal_con_ruido)
plt.title('Señal con Ruido Gaussiano')
plt.subplot(4, 1, 3)
plt.plot(tiempo, señal_filtrada)
plt.title('Señal Reconstruida por un Filtro Pasa Bajas')
plt.subplot(4, 1, 4)
plt.plot(tiempo, señal_reconstruida_7)
plt.title('Señal Reconstruida en la septima Derivada ')
plt.tight_layout()
plt.show()


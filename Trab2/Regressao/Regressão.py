import pandas as pd
import matplotlib.pyplot as plt
import os

# Carregar os dados do arquivo CSV
data = pd.read_csv(f'{os.getcwd()}/Trab2/Regressao/aerogerador.csv')

# Separar os dados em variáveis independentes (velocidade do vento) e dependentes (potência gerada)
velocidade_vento = data['Velocidade do Vento']
potencia_gerada = data['Potência do Gerador']

# Criar o gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(velocidade_vento, potencia_gerada, color='blue', alpha=0.5)
plt.title('Gráfico de Dispersão - Velocidade do Vento vs. Potência Gerada')
plt.xlabel('Velocidade do Vento (m/s)')
plt.ylabel('Potência Gerada (kW)')
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def Main():
    data = pd.read_csv(f'{os.getcwd()}/Trab2/Regressao/aerogerador.csv')

    # PlotarValores(data)

    eqmMQOTradicional = RodarMetodo(1000, data, MQOTradicional)
    print("MQO Tradicional Finalizado")
    
    eqmMQORegularizado = RodarMetodo(1000, data, MQORegularizado)
    print("MQO Regularizado Finalizado")

    eqmMVO = RodarMetodo(1000, data, MVO)
    print("MVO Finalizado")

    Metodos = ['MQO Tradicional', 'MQO Regularizado', 'MVO']
    
    medias = []
    medias.append(np.mean(eqmMQOTradicional))
    medias.append(np.mean(eqmMQORegularizado))
    medias.append(np.mean(eqmMVO))

    desvios = []
    desvios.append(np.std(eqmMQOTradicional))
    desvios.append(np.std(eqmMQORegularizado))
    desvios.append(np.std(eqmMVO))

    maiores = []
    maiores.append(np.max(eqmMQOTradicional))
    maiores.append(np.max(eqmMQORegularizado))
    maiores.append(np.max(eqmMVO))

    menores = []
    menores.append(np.min(eqmMQOTradicional))
    menores.append(np.min(eqmMQORegularizado))
    menores.append(np.min(eqmMVO))


    dados = {'Método': Metodos, 'Média': medias, 'Desvio Padrão': desvios, 'Maior': maiores, 'Menor': menores}

    # Criando um DataFrame a partir do dicionário
    df = pd.DataFrame(dados)

    nome_arquivo = 'eqms.csv'
    df.to_csv(nome_arquivo, index=False)



def RodarMetodo (num, data, method):
    eqms = []
    for i in range(num):
        data_train, data_test = SplitData(data)
        coeficientes = method(data_train)

        velocidade_vento = data_test['Velocidade do Vento']
        potencia_gerada = data_test['Potência do Gerador']

        potencias_resultantes = [coeficientes[0] + coeficientes[1] * v_v for v_v in velocidade_vento]

        eqm = calcular_eqm(potencia_gerada, potencias_resultantes)
        eqms.append(eqm)
    return eqms

def calcular_eqm(y_true, y_pred):
    # Verifica se os tamanhos dos arrays são iguais
    if len(y_true) != len(y_pred):
        raise ValueError("Os tamanhos dos arrays y_true e y_pred devem ser iguais.")

    # Calcula o EQM
    eqm = sum((y_true - y_pred) ** 2) / len(y_true)

    return eqm
            

def SplitData(data):
    # Embaralha os dados de forma aleatória
    data = data.sample(frac=1).reset_index(drop=True)

    # Calcula o índice para dividir os dados (80% para treinamento e 20% para teste)
    split_index = int(0.8 * len(data))

    # Divide os dados em conjuntos de treinamento e teste
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    # Exibe o número de amostras em cada conjunto
    return [train_data, test_data]

def PlotarValores(data):

    # Separar os dados em variáveis independentes (velocidade do vento) e dependentes (potência gerada)
    velocidade_vento = data['Velocidade do Vento']
    potencia_gerada = data['Potência do Gerador']

    # Criar o gráfico de dispersão
    # plt.figure(figsize=(8, 6))
    plt.scatter(velocidade_vento, potencia_gerada, color='blue', alpha=0.5)
    plt.title('Gráfico de Dispersão - Velocidade do Vento vs. Potência Gerada')
    plt.xlabel('Velocidade do Vento (m/s)')
    plt.ylabel('Potência Gerada (kW)')
    plt.grid(True)
    plt.show()

def MQOTradicional(data):
    # Extrair variáveis regressoras e variável observada
    variaveis_regressoras = data.drop(columns=['Potência do Gerador'])  # Todas as colunas exceto 'Potência Gerada (kW)'
    variavel_observada = data['Potência do Gerador']  # Apenas a coluna 'Potência Gerada (kW)'


    # Converter para matriz e vetor do NumPy
    matriz_regressoras = variaveis_regressoras.to_numpy()
    vetor_observado = variavel_observada.to_numpy().reshape(-1, 1) 

    x = matriz_regressoras
    y = vetor_observado

    x.shape = (len (x),1)
    y.shape = (len(y),1)

    plt.scatter(x,y,color='blue', alpha=0.5)
    plt.title('Gráfico MQO Tradicional - Velocidade do Vento vs. Potência Gerada')
    plt.xlabel('Velocidade do Vento (m/s)')
    plt.ylabel('Potência Gerada (kW)')

    X = np.concatenate((np.ones ((len(x),1)),x), axis=1)

    B = np.linalg.pinv(X.T@X)@X.T@y

    # print("Função resultante MQOTradicional:")
    # print(f'y = {B[0][0]} + ({B[1][0]} * x)')

    x_axis = np.linspace(0,20, 500)
    x_axis.shape = (len(x_axis),1)
    ones = np.ones((len(x_axis),1))
    X_new = np.concatenate((ones,x_axis),axis=1)
    Y_pred = X_new @ B

    plt.plot(x_axis,Y_pred, color="red", linestyle='--')
    plt.grid(True)
    # plt.show()

    return [B[0][0], B[1][0]]

def MQORegularizado(data, alpha=10):
    variaveis_regressoras = data.drop(columns=['Potência do Gerador'])  # Todas as colunas exceto 'Potência Gerada (kW)'
    variavel_observada = data['Potência do Gerador']  # Apenas a coluna 'Potência Gerada (kW)'


    # Converter para matriz e vetor do NumPy
    matriz_regressoras = variaveis_regressoras.to_numpy()
    vetor_observado = variavel_observada.to_numpy().reshape(-1, 1) 

    x = matriz_regressoras
    y = vetor_observado

    x.shape = (len (x),1)
    y.shape = (len(y),1)

    plt.scatter(x, y, color='blue')
    plt.title('Gráfico MQO Regularizado - Velocidade do Vento vs. Potência Gerada')
    plt.xlabel('Velocidade do Vento (m/s)')
    plt.ylabel('Potência Gerada (kW)')

    X = np.concatenate((np.ones((len(x), 1)), x), axis=1)

    # Coeficientes da regressão com penalização
    B = np.linalg.inv(X.T @ X + alpha * np.identity(X.shape[1])) @ X.T @ y
    # print("Função resultante MQORegularizado:")
    # print(f'y = {B[0][0]} + ({B[1][0]} * x)')

    x_axis = np.linspace(0, 20, 500)
    x_axis.shape = (len(x_axis), 1)
    ones = np.ones((len(x_axis), 1))
    X_new = np.concatenate((ones, x_axis), axis=1)
    Y_pred = X_new @ B

    plt.legend()
    plt.plot(x_axis, Y_pred, color="red", linestyle='--', label=f'λ: {alpha:.2f}')
    plt.grid(True)
    # plt.show()

    return [B[0][0], B[1][0]]

def MVO(data):
    variaveis_regressoras = data.drop(columns=['Potência do Gerador'])  # Todas as colunas exceto 'Potência Gerada (kW)'
    variavel_observada = data['Potência do Gerador']  # Apenas a coluna 'Potência Gerada (kW)'

    # Converter para matriz e vetor do NumPy
    matriz_regressoras = variaveis_regressoras.to_numpy()
    vetor_observado = variavel_observada.to_numpy().reshape(-1, 1)

    x = matriz_regressoras
    y = vetor_observado

    x.shape = (len(x), 1)
    y.shape = (len(y), 1)

    plt.scatter(x, y, color='blue')
    plt.title('Gráfico da Média de Valores Observáveis - Veloc do Vento vs. Pot Gerada')
    plt.xlabel('Velocidade do Vento (m/s)')
    plt.ylabel('Potência Gerada (kW)')

    # Calcula a média dos valores observados
    media_y = np.mean(y)

    # print("Função resultante MQOTradicional:")
    # print(f'y = {media_y} + (0 * x)')

    # Plot da linha horizontal representando a média dos valores observados
    plt.axhline(y=media_y, color='red', linestyle='--', label=f'Média dos valores observados: {media_y:.2f}')

    plt.legend()
    plt.grid(True)
    # plt.show()

    return [0, media_y]


Main()


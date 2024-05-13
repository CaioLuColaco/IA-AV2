from Trab1.DomCont.AlgoritmoGenetico import AlgoritmoGenetico
import numpy as np
import pandas as pd

def main():
    max_generation = 100
    nd = 20
    N = 100
    pr = 0.85
    pm = 0.01
    restricoes_dominio = (-10, 10)

    algoritmo_genetico = AlgoritmoGenetico(max_generation, fitness, nd, N, pr, pm, restricoes_dominio)

    algoritmo_genetico.geracoes()

    maioresAptidoes = algoritmo_genetico.melhor
    menoresAptidoes = algoritmo_genetico.menor
    mediasAptidoes = algoritmo_genetico.media
    desviosAptidoes = algoritmo_genetico.desvio_padrao

    # Criando um dicionário com os dados
    dados = {'Rodada': range(1, len(maioresAptidoes) + 1), 'Maior Aptidão': maioresAptidoes, 'Menor Aptidão': menoresAptidoes, 'Media Aptidão': mediasAptidoes, 'Desvios Padrão': desviosAptidoes}

    # Criando um DataFrame a partir do dicionário
    df = pd.DataFrame(dados)

    nome_arquivo = 'aptidoesDomCont.csv'
    df.to_csv(nome_arquivo, index=False)

    # Imprimindo o DataFrame
    # print(df)



def fitness(x, y):
    return rastrigin(x, y) + 1

def rastrigin(x, y):
    A = 10
    return A * 2 + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y))


main()

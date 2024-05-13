import csv
import random
import math
import numpy as np

class Ponto:
    def __init__(self, coordenadas, grupo):
        self.coordenadas = coordenadas
        self.grupo = grupo

class Individuo:
    def __init__(self, sequencia):
        self.sequencia = sequencia
        self.fitness = None

def ler_pontos_do_csv(nome_arquivo):
    pontos = []
    with open(nome_arquivo, 'r') as arquivo:
        leitor_csv = csv.reader(arquivo)
        for linha in leitor_csv:
            coordenadas = tuple(map(float, linha[:3]))
            grupo = int(linha[3])
            ponto = Ponto(coordenadas, grupo)
            pontos.append(ponto)
    return pontos

def calcular_distancia(ponto1, ponto2):
    return math.sqrt((ponto2.coordenadas[0] - ponto1.coordenadas[0])**2 + 
                     (ponto2.coordenadas[1] - ponto1.coordenadas[1])**2 + 
                     (ponto2.coordenadas[2] - ponto1.coordenadas[2])**2)

def custo_total(sequencia, pontos):
    custo = 0
    for i in range(len(sequencia) - 1):
        ponto_atual = pontos[sequencia[i]]
        proximo_ponto = pontos[sequencia[i+1]]
        custo += calcular_distancia(ponto_atual, proximo_ponto)
    custo += calcular_distancia(pontos[sequencia[-1]], pontos[0])
    return custo

def inicializar_populacao(tamanho_populacao, pontos):
    populacao = []
    for _ in range(tamanho_populacao):
        sequencia = []
        grupos_visitados = set()
        pontos_disponiveis = pontos.copy()
        
        for grupo in range(5):
            pontos_grupo = [ponto for ponto in pontos_disponiveis if ponto.grupo == grupo]
            ponto_selecionado = random.choice(pontos_grupo)
            sequencia.append(pontos.index(ponto_selecionado))
            grupos_visitados.add(grupo)
            pontos_disponiveis.remove(ponto_selecionado)
        
        random.shuffle(pontos_disponiveis)
        
        sequencia.extend([pontos.index(ponto) for ponto in pontos_disponiveis])
        
        individuo = Individuo(sequencia)
        populacao.append(individuo)
    
    return populacao


def selecao_torneio(populacao, k):
    pais = []
    for _ in range(len(populacao)):
        torneio = random.sample(populacao, k)
        vencedor = min(torneio, key=lambda x: x.fitness)
        pais.append(vencedor)
    return pais


def recombinação_dois_pontos(pai1, pai2):
    ponto1, ponto2 = sorted(random.sample(range(len(pai1.sequencia)), 2))
    filho = [-1] * len(pai1.sequencia)
    filho[ponto1:ponto2] = pai1.sequencia[ponto1:ponto2]

    index = ponto2
    for gene in pai2.sequencia:
        if gene not in filho:
            filho[index] = gene
            index = (index + 1) % len(pai1.sequencia)

    return Individuo(filho)

def mutacao(individuo):
    if random.random() < 0.05:
        ponto1, ponto2 = sorted(random.sample(range(len(individuo.sequencia)), 2))
        individuo.sequencia[ponto1], individuo.sequencia[ponto2] = individuo.sequencia[ponto2], individuo.sequencia[ponto1]
    return individuo


def algoritmo_genetico(tamanho_populacao, max_geracoes, nome_arquivo):
    pontos = ler_pontos_do_csv(nome_arquivo)
    populacao = inicializar_populacao(tamanho_populacao, pontos)
    
    custos_rodada = []

    for individuo in populacao:
        individuo.fitness = custo_total([0] + individuo.sequencia + [0], pontos)
    
    contador_geracoes = 0
    
    for geracao in range(max_geracoes):
        nova_populacao = []

        elites = sorted(populacao, key=lambda x: x.fitness)[:5]
        nova_populacao.extend(elites)

        while len(nova_populacao) < tamanho_populacao:
            pais = selecao_torneio(populacao, 5)
            pai1, pai2 = random.sample(pais, 2)
            filho = recombinação_dois_pontos(pai1, pai2)
            filho = mutacao(filho)
            filho.fitness = custo_total([0] + filho.sequencia + [0], pontos)
            nova_populacao.append(filho)

        populacao = nova_populacao

        custos_rodada.append([individuo.fitness for individuo in populacao])

        contador_geracoes += 1
        if contador_geracoes >= max_geracoes:
            break

    melhor_individuo = min(populacao, key=lambda x: x.fitness)
    melhor_caminho = [0] + melhor_individuo.sequencia + [0]
    melhor_custo = melhor_individuo.fitness
    return melhor_caminho, melhor_custo, custos_rodada

def calcular_estatisticas_custo(custos_todas_rodadas):
    menor_custo = min([min(rodada) for rodada in custos_todas_rodadas])
    maior_custo = max([max(rodada) for rodada in custos_todas_rodadas])
    media_custo = np.mean(custos_todas_rodadas)
    desvio_padrao_custo = np.std(custos_todas_rodadas)
    return menor_custo, maior_custo, media_custo, desvio_padrao_custo

def salvar_resultados_csv(nome_arquivo, resultados):
    with open(nome_arquivo, 'a', newline='') as arquivo:
        escritor_csv = csv.writer(arquivo)
        escritor_csv.writerow(resultados)

for i in range(100):
    _, _, custos_rodada = algoritmo_genetico(tamanho_populacao=100, max_geracoes=500, nome_arquivo='caixeiro-grupos.csv')

    estatisticas = calcular_estatisticas_custo(custos_rodada)

    salvar_resultados_csv('resultados_caixeiro_grupos.csv', estatisticas)

print("Processo concluído. Resultados salvos em 'resultados_caixeiro_grupos.csv'.")

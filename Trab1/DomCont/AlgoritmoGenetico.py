from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

class AlgoritmoGenetico:
    """Construtor do algoritmo genético.

    ### Parameters
    1. max_generation : int
        - Quantidades de gerações para convergência do algoritmo.
    2. fitness : Callable[[float,float], float]
        - Uma função de aptidão. Para o presente exemplo, 
            esta função possui dois argumentos de tipo float e um retorno de tipo float.
    3. nd : int        
        - Quantidade de bits presentes para cada parâmetro p.
    4. N : int        
        - Quantidade de indivíduos na população.
    5. pr : float        
        - Probabilidade de recombinação. Deve ser um número real no intervalo [0,1]. Contudo, recomenda-se um alto valor ex: 0.85
    6. pm : float        
        - Probabilidade de mutação. Deve ser um número real no intervalo [0,1]. Contudo, recomenda-se um baixo valor ex: 0.01
    7. restricoes_dominio : tuple        
        - Tupla contendo os limites da restrição do problema. No presente exemplo, para as duas dimensões, os limites inferior e superior serão os mesmos.
      
    ### Returns
    - Sem retornos no construtor.

    
    """
    def __init__(self,max_generation:int,fitness: Callable[[float,float], float],nd:int,N:int,pr:float,pm:float,restricoes_dominio:tuple) -> None:
        self.max_generation = max_generation
        self.l = restricoes_dominio[0]
        self.u = restricoes_dominio[1]
        self.fitness = fitness
        self.p = 2 # Quantidade de parâmetros do problema. Para o presente exemplo, p sempre é 2.
        self.nd = nd
        self.N = N
        self.pr = pr
        self.pm = pm
        self.P = None #atributo associado a população de cada geração.
        self.S = None #atributo associado ao grupo de selecionados para o processo de recombinação.
        self.aptidoes = np.zeros(self.N) #vetor com aptidões de cada indivíduo, que será atualizado a cada geração.
        self.total_aptidoes = 0 #atributo associado a soma de aptidões de cada geração.
        self.melhor = [] #lista de melhor resultado a cada geração.
        self.media = [] #lista com a média de aptidões em cada geração.
        self.menor = [] #lista com o menor de aptidões em cada geração.
        self.desvio_padrao = [] #lista com o desvio padrão das aptidões em cada geração.

    
    def inicializar_plot(self):
        '''
        Método utilizado apenas para inicializar o plot a cada geração.
        Não faz parte do algoritmo genético de fato, ou seja, poderia estar presente em outra classe (uma utilitária).
        '''
        x_axis = np.linspace(self.l,self.u,1000)
        X,Y = np.meshgrid(x_axis,x_axis)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        Z = self.fitness(X,Y)
        ax.plot_surface(X,Y,Z,cmap='gray',edgecolor='k',
                        linewidth=0.5,alpha=.5)
        return ax
        

    def gerar_populacao(self):
        """
            Método que inicializa uma população de individiduos de dimensão (N x p*nd).
            Tal inicialização é aleatória e segue distribuição uniforme.
        """
        return np.random.randint(low=0,high=2,size=(self.N,self.p*self.nd))

    def phi(self,x:np.ndarray) -> float:
        """
            Método que decodifica a sequência binária para uma representação real.

            ### Parameters
                1. x : np.ndarray        
                    - Um numpy array de shape (nd x p,), que representa um único indivíduo na população.
                
            ### Returns
                - float
                    - Retorna um número real decodificado da representação binária
        """
        dec = 0
        for i in range(len(x)):
            dec += x[len(x)-1-i]* 2**i

        return self.l + (self.u-self.l)/(2**self.nd-1)*dec

    def roleta(self):
        i = 0
        soma = self.aptidoes[i]/self.total_aptidoes
        r = np.random.uniform()
        while soma < r:
            i+=1
            soma += self.aptidoes[i]/self.total_aptidoes
        return self.P[i,:]

    def selecao(self):

        S = np.empty((0,self.nd*self.p))
        for i in range(self.N):
            s = self.roleta()
            S = np.concatenate((
                S,
                s.reshape(1,self.nd*self.p)
            ))
        return S


    def calcular_aptidoes(self):
        lb = []
        ax = self.inicializar_plot()
        for i in range(self.N):
            x,y = self.phi(self.P[i,0:self.nd]),self.phi(self.P[i,self.nd:])
            self.aptidoes[i] = self.fitness(x,y)
            ax.scatter(x,y,self.aptidoes[i],s=90,color='k',edgecolor='blue')
            lb.append(f"ind {i}")
        self.total_aptidoes = np.sum(self.aptidoes)

        self.melhor.append(np.max(self.aptidoes))

        self.menor.append(np.min(self.aptidoes))

        desvio = calcular_desvio_padrao(self.aptidoes)
        self.desvio_padrao.append(desvio)

        self.media.append(np.mean(self.aptidoes))
        # plt.show()
        # prob = self.aptidoes/self.total_aptidoes
        # plt.pie(prob,labels=lb)
        # plt.show()

    def recombinacao(self):
        R = np.empty((0,self.nd*self.p))

        for i in range(0,self.N,2):
            x1 = self.S[i,:]
            x2 = self.S[i+1,:]
            x1_t = np.copy(x1)
            x2_t = np.copy(x2)
            if(np.random.uniform()<=self.pr):
                m = np.zeros(self.p*self.nd)
                xi = np.random.randint(0,self.p*self.nd-1)
                m[xi+1:] = 1
                correspondencia = m[:]==1
                x1_t[correspondencia] = x2[m[:]==1]
                x2_t[m[:]==1] = x1[m[:]==1]

            
            R = np.concatenate((
                R,
                x1_t.reshape(1,self.p*self.nd),
                x2_t.reshape(1,self.p*self.nd),
            ))
        return R

    def __toggle(self,x):
        return 1 if x==0 else 0
                
    def mutacao(self):
        for i in range(self.N):
            for j in range(self.nd*self.p):
                if np.random.uniform()<= self.pm:
                    self.P[i,j] = self.__toggle(self.P[i,j])

        
    def geracoes(self):

        self.P = self.gerar_populacao()
        for i in range(self.max_generation):
            self.calcular_aptidoes()
            self.S = self.selecao()
            self.P = self.recombinacao()
            self.mutacao()

def calcular_desvio_padrao(array):
    media = np.mean(array)
    
    dif_quad = [(x - media) ** 2 for x in array]

    variancia = np.mean(dif_quad)
    
    desvio_padrao = np.sqrt(variancia)
    
    return desvio_padrao

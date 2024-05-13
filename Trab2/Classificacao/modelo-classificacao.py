import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

Data = pd.read_csv('EMGDataset.csv', header=None,names=['Sensor1', 'Sensor2', 'Classificacao'])

X = Data.values
N,p = X.shape

neutro = np.tile(np.array([[1,-1,-1,-1,-1]]),(1000,1)) 
sorrindo = np.tile(np.array([[-1,1,-1,-1,-1]]),(1000,1)) 
aberto = np.tile(np.array([[-1,-1,1,-1,-1]]),(1000,1)) 
surpreso = np.tile(np.array([[-1,-1,-1,1,-1]]),(1000,1)) 
rabugento = np.tile(np.array([[-1,-1,-1,-1,1]]),(1000,1)) 
Y = np.tile(np.concatenate((neutro,sorrindo,aberto,surpreso,rabugento)),(10,1))


def bestAlpha(rounds, X, Y):
    alphaValues =  np.arange(0.1, 1.01, 0.1) 
    bestAlpha = 1
    maxValue = -1

    for currentAlpha in alphaValues:

        accuracies_alpha = []

        for round in range(rounds):
            indexRandom = np.random.permutation(N)
            indexPercentage = int(N*.8)

            X_randomize = X[indexRandom,:]
            Y_randomize = Y[indexRandom,:]

            X_train = X_randomize[0: indexPercentage,:] 
            Y_train = Y_randomize[0: indexPercentage,:]
            X_test =  X_randomize[indexPercentage: N,:] 
            Y_test =  Y_randomize[indexPercentage: N,:]

            model_Tikhonov_alpha = np.linalg.inv((X_train.T @ X_train) + currentAlpha * np.identity((X_train.T @ X_train).shape[0]))@ X_train.T @ Y_train

            Y_predict_alpha = X_test @ model_Tikhonov_alpha

            discriminant_predict_alpha = np.argmax(Y_predict_alpha, axis=1)
            discriminant_test = np.argmax(Y_test, axis=1)
            accuracy_Tikhonov_alpha = accuracy_score(discriminant_predict_alpha, discriminant_test)

            accuracies_alpha.append(accuracy_Tikhonov_alpha)
        
        if(np.mean(accuracies_alpha) > maxValue):
            maxValue = np.mean(accuracies_alpha)
            best_alpha = currentAlpha

    return best_alpha

    
def eucledian_distance(x1, x2):
    """Eucldian distance"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_classifier(X_train, y_train, X_test, k):
    y_pred = []
    for i in range(len(X_test)):
        print(i)
        distancies = [eucledian_distance(X_train[j], X_test[i]) for j in range(len(X_train))]
        index_neighbors = np.argsort(distancies)[:k]
        neighbors = [y_train[idx] for idx in index_neighbors]
        
        # Encontre a classe mais frequente usando a função numpy unique
        classes, counts = np.unique(neighbors, return_counts=True)
        frequent_class = classes[np.argmax(counts)]
        
        y_pred.append(frequent_class)
    return np.array(y_pred)

def dmc_classifier(X_train, Y_train, X_test):
    centroids = []
    for label in np.unique(Y_train):
        class_labels = X_train[Y_train == label]
        centroid = np.mean(class_labels, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    Y_pred = np.argmin(np.linalg.norm(X_test[:, np.newaxis] - centroids, axis=2), axis=1)
    return Y_pred

def accuracy(X_Test, Y_test, model):
    Y_predict = X_Test @ model

    discriminant_predict = np.argmax(Y_predict, axis=1)
    discriminant_test = np.argmax(Y_test, axis=1)
    accuracy_model = accuracy_score(discriminant_predict, discriminant_test)

    return accuracy_model

rounds = 100

best_alpha = bestAlpha(rounds, X , Y)

accuracy_OLS_rounds = []
accuracy_Tikhonov_rounds = []
accuracy_KNN_rounds = []
accuracy_KNN_function_rounds = []
accuracy_DMC_rounds = []

interceptor = np.ones((X.shape[0] , 1)) 
X = np.concatenate((interceptor , X),axis=1)

for rodada in range(rounds):
    indexRandom = np.random.permutation(N)
    indexPercentage = int(N*.8)

    #Embaralhar dados
    X_randomize = X[indexRandom,:]
    Y_randomize = Y[indexRandom,:]

    X_train = X_randomize[0: indexPercentage,:] 
    Y_train = Y_randomize[0: indexPercentage,:]
    X_test =  X_randomize[indexPercentage: N,:] 
    Y_test =  Y_randomize[indexPercentage: N,:]

    #OLS Model
    model_OLS = np.linalg.pinv(X_train.T@X_train)@X_train.T@Y_train
    accuracy_OLS = accuracy(X_test, Y_test, model_OLS)
    accuracy_OLS_rounds.append(accuracy_OLS)

    #Tikhonov Model
    model_Tikhonov = np.linalg.inv((X_train.T @ X_train) + best_alpha * np.identity((X_train.T @ X_train).shape[0]))@ X_train.T @ Y_train
    accuracy_Tikhonov = accuracy(X_test, Y_test, model_Tikhonov)
    accuracy_Tikhonov_rounds.append(accuracy_Tikhonov)

    # KNN Model (LIB)
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    
    Y_predict_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(Y_test, Y_predict_knn)
    accuracy_KNN_rounds.append(accuracy_knn)

    #DMC Model
    dmc = dmc_classifier(X_train, np.argmax(Y_train, axis=1), X_test)
    discriminant_test = np.argmax(Y_test,axis=1)
    accuracy_dmc = accuracy_score(dmc, discriminant_test)
    accuracy_DMC_rounds.append(accuracy_dmc)

# Accuracy for OLS
mean_accuracy_OLS = np.mean(accuracy_OLS_rounds)
std_accuracy_OLS = np.std(accuracy_OLS_rounds)
minvalue_accuracy_OLS = min(accuracy_OLS_rounds)
maxvalue_accuracy_OLS = max(accuracy_OLS_rounds)
mode_accuracy_OLS = stats.mode(accuracy_OLS_rounds)
print(mean_accuracy_OLS, std_accuracy_OLS, minvalue_accuracy_OLS, maxvalue_accuracy_OLS, mode_accuracy_OLS[0])

# Accuracy for Tikhonov
mean_accuracy_Tikhonov = np.mean(accuracy_Tikhonov_rounds)
std_accuracy_Tikhonov = np.std(accuracy_Tikhonov_rounds)
minvalue_accuracy_Tikhonov = min(accuracy_Tikhonov_rounds)
maxvalue_accuracy_Tikhonov = max(accuracy_Tikhonov_rounds)
mode_accuracy_Tikhonov = stats.mode(accuracy_Tikhonov_rounds)
print(mean_accuracy_Tikhonov, std_accuracy_Tikhonov, minvalue_accuracy_Tikhonov, maxvalue_accuracy_Tikhonov, mode_accuracy_Tikhonov[0])

# Accuracy for KNN
mean_accuracy_KNN = np.mean(accuracy_KNN_rounds)
std_accuracy_KNN = np.std(accuracy_KNN_rounds)
minvalue_accuracy_KNN = min(accuracy_KNN_rounds)
maxvalue_accuracy_KNN = max(accuracy_KNN_rounds)
mode_accuracy_KNN = stats.mode(accuracy_KNN_rounds)
print(mean_accuracy_KNN, std_accuracy_KNN, minvalue_accuracy_KNN, maxvalue_accuracy_KNN, mode_accuracy_KNN[0])

# Accuracy for DMC
mean_accuracy_DMC = np.mean(accuracy_DMC_rounds)
std_accuracy_DMC = np.std(accuracy_DMC_rounds)
minvalue_accuracy_DMC = min(accuracy_DMC_rounds)
maxvalue_accuracy_DMC = max(accuracy_DMC_rounds)
mode_accuracy_DMC = stats.mode(accuracy_DMC_rounds)
print(mean_accuracy_DMC, std_accuracy_DMC, minvalue_accuracy_DMC, maxvalue_accuracy_DMC, mode_accuracy_DMC[0])

# Valores para cada métrica
means = [mean_accuracy_OLS, mean_accuracy_Tikhonov, mean_accuracy_KNN, mean_accuracy_DMC]
stds = [std_accuracy_OLS, std_accuracy_Tikhonov, std_accuracy_KNN, std_accuracy_DMC]
minValues = [minvalue_accuracy_OLS, minvalue_accuracy_Tikhonov, minvalue_accuracy_KNN, minvalue_accuracy_DMC]
maxValues = [maxvalue_accuracy_OLS, maxvalue_accuracy_Tikhonov, maxvalue_accuracy_KNN, maxvalue_accuracy_DMC]
modes = [mode_accuracy_OLS[0], mode_accuracy_Tikhonov[0], mode_accuracy_KNN[0], mode_accuracy_DMC[0]]

largura_barras = 0.2

models = ['OLS', 'Tikhonov', 'KNN', 'DMC']

# Posições das barras no eixo x
positions_m1 = np.arange(len(models))
positions_m2 = [x + largura_barras for x in positions_m1]
positions_m3 = [x + largura_barras for x in positions_m2]
positions_m4 = [x + largura_barras for x in positions_m3]
positions_m5 = [x + largura_barras for x in positions_m4]

# Criar o gráfico de barras
plt.bar(positions_m1, means, largura_barras, label='Média')
plt.bar(positions_m2, stds, largura_barras, label='Desvio Padrão')
plt.bar(positions_m3, minValues, largura_barras, label='Menor Valor')
plt.bar(positions_m4, maxValues, largura_barras, label='Maior Valor')
plt.bar(positions_m5, modes, largura_barras, label = 'Moda')

# Adicionar detalhes ao gráfico
plt.xlabel('Modelos')
plt.ylabel('Valores')
plt.title('Comparação das Métricas para os Modelos')
plt.xticks(positions_m2, models)
plt.legend()

# Mostrar o gráfico
plt.tight_layout()
plt.show()

# Identificar a expressão correspondente a cada bloco de 10.000 observações
expressions = ['Neutro', 'Sorriso', 'Sobrancelhas levantadas', 'Surpreso', 'Rabugento']
repeated_expressions = [expression for expression in expressions for _ in range(10000)]

Data['Expressao'] = repeated_expressions

colors = plt.cm.viridis.colors

expression_colors = [
    colors[i * len(colors) // len(expressions)] for i in range(len(expressions))]


# Plotar cada expressão facial separadamente para adicionar legenda
if(True):
    for i, expression in enumerate(expressions):
        expressions_data = Data[Data['Expressao'] == expression]
        plt.scatter(expressions_data['Sensor1'], expressions_data['Sensor2'],
                    label=expression, color=[expression_colors[i]])

    plt.xlabel('Sensor1')
    plt.ylabel('Sensor2')
    plt.title('Gráfico de Dispersão dos Sensores por Expressão Facial')
    plt.legend()
    plt.show()
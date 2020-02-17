#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:23:36 2018

Info: PREENCHER


@author: cesar
"""
########## 

#Dados obtidos do site yahoo finanças
#Dados da PETR4 de 1 janeiro de 2013 até  de janeiro de 2018
# Objetivo prever valor ação de valores de janeiro de 2018
from keras.models import Sequential
from sklearn.metrics import r2_score #Novo#
from keras.layers import Dense, Dropout, LSTM # LSTM adicionado na aula 70
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.layers import Bidirectional

# classe para facilitar como mostrar
class Camada:
    def __init__(self, dropout, neuronio, return_sequences, tipo = 1): #init construtor
        self.dropout = dropout #self - vai receber o que passou de parametro
        self.neuronio = neuronio
        self.return_sequences = return_sequences
        self.tipo = tipo

    def __repr__(self):
        return "("+str(self.neuronio)+")"
####################################################################################
# Preencher o quadro abaixo manualmente        
########################################
#Inicio, fim, intervalo
#epocas = [n for n in range(100, 500, 100)]
epocas = [100]

#Opções de batchs
batchSizes = [32, 64]
#Quantidade de numeros para prever um valor
anteriores = [400] ### Considerar

#Definir numero de camadas ocultas 
ocultas = [4]

#Quantidade de células de memória
neuronios = [100]

#Tipode de LSM: Hidder e Bidirectional
tipos = [1, 2]

#Nome de arquivos associados a este arquivo
file1 = 'outono20112016_90_10.h5'
file2 = 'outono20112016_90_10.txt'

configuracoes=[]
####################################################################################
for oculta in ocultas:
    for neuronio in neuronios:
        for tipo in tipos:
            last =  [Camada(0.3, neuronio, False, tipo)]
            novo =  [Camada(0.3, neuronio, True, tipo)] * (oculta - 1) #Esse -1 é por conta da -ultima camada oculta deve retornar False para termo return
            cocultas = novo + last
            configuracoes.append(cocultas)

##########

base = pd.read_csv('outono2016_2016_treinamento.csv')
file_teste = 'outono2016_2016_teste.csv'

#############################
coluna_prever = 1 #Considerando que começa com 1, já no python começa com zero
nome_coluna = base.columns[(coluna_prever-1)]
print("Coluna que deseja-se prever:",(coluna_prever), "- Nome da coluna:", nome_coluna)
#############################
#Apagar registros NaN
base = base.dropna()

#Definir quantidade de colunas. Obs: vai usar todas do arquivo
colunas = len(base.columns)

#Definir quantidade de linhas do arquivo
linhas = len(base.index)

#Separar qual vai ser o valor para fazer previsão, será da coluna 1 a 2, serão os previsores
base_treinamento = base.iloc[:,0:colunas].values # Coluna 1,2,3

#Normalizar dados
#Esse tipo de rede neural pode ficar lenta se usar valores reais, fazer escala de 0 a 1
normalizador = MinMaxScaler(feature_range=(0,1))

# Variável com a base com valores normalizados entre 0 e 1
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

# Ficar na mesma escala de base_treinamento
normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(base_treinamento[:,(coluna_prever-1):coluna_prever])# #!!!! Escolher coluna para prever!!! #

########

# Para fazer previsão temporal é necessário passar para rede neural em outro formato
# É necessário preparar a base de dados
# Para trabalhar com série temporal é necessário primeiramente definir o intervalo de tempo
# É necessário ter "previsores" e "preço real"
# É necessário ter dados anteriores (pelo menos 4) para uma data que se deseja prever
# É necessário ter coluna com informação de data

########

# Codificação para construir base de dados
# Utilizar 90 dias anteriores para prever valor da ação
for anterior in anteriores:
    previsores = [] #lista
    preco_real = [] #lista
    
    # Deve começar a tentar prever a partir da posição 90, pois os primeiros valores são para tentar prever. 1242 é o numero de registros
    for i in range(anterior, linhas): # append é para adicionar dado na lista
        previsores.append(base_treinamento_normalizada[i-anterior:i, 0:colunas])# i = 0,1,2,3... . vai até i, indice 0 a 2
        # preenche preço real
        preco_real.append(base_treinamento_normalizada[i, (coluna_prever-1)]) # i = 90 , preço das 90 datas anteriores, 0 = primeira coluna
    
    #transformar dados para tipo numpy, necessário para passar para rede neural
    previsores, preco_real = np.array(previsores), np.array(preco_real)
    
    #############################################################################################
    
    ########
    
    #Teste em outra base
    base_teste = pd.read_csv(file_teste)
    #Extrair somente variável Open, pegando apenas primeira coluna
    preco_real_teste = base_teste.iloc[:,(coluna_prever-1):coluna_prever].values #!!!! Escolher coluna para prever!!! #
    
    #Definir quantidade de linhas do arquivo teste
    linhas2 = len(base_teste.index)
    
    ########
    
    #Agora temos mais atributos
    frames = [base, base_teste]
    
    #Vamos precisar dos 90 preços anteriores, para isso podemos concatenar
    base_completa = pd.concat(frames) 
    
    #Apagar coluna "Dates", não é usado
    #base_completa = base_completa.drop('Date', axis = 1)
    
    entradas = base_completa[len(base_completa) - len(base_teste) - anterior:].values #.values converte para formato numpy array, -90 para indicar onde começa a checar
    
    #Colocar dados na mesma escala
    entradas = normalizador.transform(entradas)
    
    #Somatorio de anterior(quantidade de dias) + linhas2(arquivo teste)
    total = anterior + linhas2
    
    X_teste = [] # lista fazia
    for i in range(anterior, total): # 90 + 22(registros) = 112
        X_teste.append(entradas[i-anterior:i, 0:colunas])#coluna 0 até 6
    X_teste = np.array(X_teste)
    #Agora temos 90 colunas e 22 registros com os 90 valores de açãoes antes de janeiro
    #############################################################################################
    
    
    for c in range(len(configuracoes)):
        configuracao = configuracoes[c]
        for epoca in epocas:
            for batch in batchSizes:
                ######### Aula 70 #########
    
                #Criar regressor
                regressor = Sequential()
    
                # Adicionar camadas
                # units = numero de células de memória, deve ser número grande para adicionar mais dimensionalidade e capturar uma tendência ao decorrer do tempo
                # return_sequences = mais uma camada LSTM, vai passar informação para frente, para outras camadas
                # input_shape = dados de entrada. shape[1], 1 -> somente 1 previsor(Open)
                lstm = LSTM(units=configuracao[0].neuronio, return_sequences=configuracao[0].return_sequences, input_shape=(previsores.shape[1], colunas))
                if(configuracao[0].tipo == 2):
                    lstm = Bidirectional(lstm)
                regressor.add(lstm)
                regressor.add(Dropout(configuracao[0].dropout))  # dropout vai zerar 30% das entradas para prevenir overfitting
                # É interessante adicionar mais camadas
                # Depois da primeira camada pode abaixar valor de units
                # input_shape uas apenas na primeira
    
                for i in range(1, len(configuracao)):
                    lstm = LSTM(units=configuracao[i].neuronio, return_sequences=configuracao[i].return_sequences)
                    if(configuracao[i].tipo == 2):
                        lstm = Bidirectional(lstm)
                    regressor.add(lstm)
                    regressor.add(Dropout(configuracao[i].dropout))  # dropout vai zerar 30% das entradas para prevenir overfitting
                #Adicionar camadas
                #units = numero de células de memória, deve ser número grande para adicionar mais dimensionalidade e capturar uma tendẽncia ao decorrer do tempo
                #return_sequences = mais uma camada LSTM, vai passar informação para frente, para outras camadas
                #input_shape = dados de entrada. shape[1], 6 -> 6 atributos (colunas) previsores
    
                #Camada de saída
                #units 1 = igual uma saída apenas
                #função de ativação, usar sigmoid, pois normalizou valores entre zero e um. Poderia usar linear também.
                regressor.add(Dense(units = 1, activation = 'sigmoid'))
    
                #RMSprop é recomendado para esse tipo de rede neural, segundo a documentação Keras
                #mean_squared_error é um cálculo mais eficiente
                #mean_absolute_error é usado na métrica para facilitar o entendimento, visualizar resultado
                regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',
                                  metrics = ['mean_absolute_error'])
    
                ########
    
                #Classe EarlyStopping = faz a parada de processameno antes, de acordo com condições
                #Monitor loss, vai verificar se a loss function vai melhorar ou não
                #Min_delta em notação científica. 10 zeros antes de 1
                #Patience = 10 = quer dizer se passar 10 épocas sem melhorar loss function ele termina o treinamento
                #Verbose mostra mensagens na tela
                #1e-10 = notação cienífica = 10 zeros antes de 1
                es = EarlyStopping(monitor =  'loss', min_delta = 1e-10, patience = 10, verbose = 1)
    
                #Reduzir taxa de aprendizagem quando uma métrica parou de melhorar
                rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, epochs = 5, verbose = 1)
    
                #Salvar modelo após cada época, salva os pesos, no final consegue pegar melhor modelo
                mcp = ModelCheckpoint(filepath = file1, monitor = 'loss', save_best_only = True, verbose = 1)
    
    
                #Deve usar pelo menos 100 epocas
                regressor.fit(previsores, preco_real, epochs = epoca, batch_size = batch, callbacks = [es, rlr, mcp])
                #TReinamento feito na própria base
                # Na próxima aula será feita com outra base
    
    
                #Agora fazer previsão
                previsoes = regressor.predict(X_teste)
                #Agora temos as respostas que a rede neural recorrente gerou
    
                #processo inverso da normalização, para facilitar a comparação de preços
                previsoes = normalizador_previsao.inverse_transform(previsoes)
                
                #
                dados = str(epoca)+'Epoca'+'.Batch'+str(batch)
    
                #
                print("-"*45)
                print(f"Estatísticas: {dados}")
                print("-"*45)
    
                #Media
                medprev = previsoes.mean()
                print(f"Media Prev = {medprev}")
    
                medreal = preco_real_teste.mean()
                print(f"Media Real = {medreal}")
    
                print("-"*45)
                #Variancia
                varprev = previsoes.var()
                print(f"Variancia Prev = {varprev}")
    
                varreal = preco_real_teste.var()
                print(f"Variancia Real = {varreal}")
    
                print("-"*45)
                #Desvio padrão
                dpprev = previsoes.std()
                print(f"Desvio P Prev = {dpprev}")
    
                dpreal = preco_real_teste.std()
                print(f"Desvio P Real = {dpreal}")
                print("-"*45)
                
                # Mean Squared Error
                mse = mean_squared_error(preco_real_teste, previsoes)
                print(f"MSE = {mse}")
                
                #Root Mean Squared Error
                rmse = sqrt(mean_squared_error(preco_real_teste, previsoes))
                print(f"RMSE = {rmse}")
                print("-"*45)
                
                #R Square - R quadrado
                rq = r2_score(preco_real_teste, previsoes)
                print(f"R Quadrado = {rq}")
                print("-"*45)
    
                print("-"*45)
        
                #### Escrita em arquivo
                
                #Gerar nome
                dados = str(epoca)+'Epoca'+'.Batch'+str(batch)
                #
                f = open(file2, 'a')
                f.write('' '\n')
                f.write('**************************' '\n')
                f.write('Dados de: ' + repr(dados) + '\n')
                f.write('--------------------------' '\n')
                f.write('Media Prev = ' + repr(medprev) + '\n')
                f.write('Media Real = ' + repr(medreal) + '\n')
                f.write('' '\n')
                f.write('Variancia Prev = ' + repr(varprev) + '\n')
                f.write('Variancia Real = ' + repr(varreal) + '\n')
                f.write('' '\n')
                f.write('DesvioPadrão Prev = ' + repr(dpprev) + '\n')
                f.write('DesvioPadrão Real = ' + repr(dpreal) + '\n')
                f.write('' '\n')
                f.write('--------------------------' '\n')
                f.write('MSE Prev = ' + repr(mse) + '\n')          
                f.write('RMSE Prev = ' + repr(rmse) + '\n')          
                f.write('--------------------------' '\n')
                
                #### Escrita em arquivo
                plt.plot(preco_real_teste, color = 'red', label = 'Temp Real')
                plt.plot(previsoes, color = 'blue', label = 'Previsões')
                plt.title('Previsão Temp')
                plt.xlabel('Tempo em dias')
                plt.ylabel('Temperatura')
                plt.legend()
                nome = 'Epoca'+str(epoca)+'.Batch'+str(batch)+'.MSE='+str(mse)+'.RQ='+str(rq)
                nome_tipo = "Stack"
                if(configuracao[0].tipo == 2):
                    nome_tipo = "Bidirectional"
                plt.savefig('v6.outono.2016-2016.'+str(len(configuracao))+'Cam.'+str(configuracao[0].neuronio)+'Neu.'+nome_tipo+'Tipo.Temp.'+nome+'.Ant.'+str(anterior)+'.png')
                #plt.show()
                plt.close()
 #Fechamento do arquivo de escrita            
f.close()

####### ORDENA o ranking por MSE e depois por RMSE, depoir RQ e gera arquivos csv
#datas = pd.read_csv("ranking.csv") 
#datas["Rank"] = datas["MSE"].rank(method ='min') 
#datas.sort_values("MSE", inplace = True) 
#datas.to_csv(r'rankingMSE.csv', index=False )

#datas = pd.read_csv("ranking.csv") 
#datas["Rank"] = datas["RMSE"].rank(method ='min') 
#datas.sort_values("RMSE", inplace = True) 
#datas.to_csv(r'rankingRMSE.csv', index=False )

#datass = pd.read_csv("ranking.csv") 
#datass["Rank"] = datass["RQ"].rank(method ='min') 
#datass.sort_values("RQ", inplace = True) 
#datass.to_csv(r'rankingRQ.csv', index=False )

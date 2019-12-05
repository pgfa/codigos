# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:18:28 2019

@author: lucas e césar
"""


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
from keras.models import load_model
from sklearn.metrics import r2_score


modelo =  ['inverno20112016_90_10.h5',
          'outono20112016_90_10.h5',
          'primavera20112016_90_10.h5', 
          'verao20112016_90_10.h5']
#nome = ['inverno20112016.csv']
nome = ['inverno2016_2016.csv', 'outono2016_2016.csv', 
        'primavera2016_2016.csv', 'verao2016_2016.csv']


for mod in modelo:
    for nom in nome:
        #carregar base
        base = pd.read_csv(nom)
        
        #INformações na tela
        print("Testando o modelo: ", mod, "em dados de:", nom)
        model = load_model(mod) # Carrrega o modelo treinado
        model.summary() # Mostra o sumario do modelo
        
        #Definir quantidade de colunas. Obs: vai usar todas do arquivo
        colunas = len(base.columns)
        print('Quantidade de variaveis (colunas) utilizadas:') # Mostra a quantidade de variaveis utilizadas no treinamento
        print(colunas)
        
        #Definir quantidade de linhas do arquivo
        linhas = len(base.index)
        
        #Separar qual vai ser o valor para fazer previsão, será da coluna 1 a 2, serão os previsores
        base_treinamento = base.iloc[:,0:colunas].values # Coluna Todas as colunas
        base_treinamento2 = base.iloc[:,0:1].values #Pega somente a coluna 1
        #coordenadas = base.iloc[400:,0:4].values #Pegar LAT e LONG
        #Normalizar dados
        #Esse tipo de rede neural pode ficar lenta se usar valores reais, fazer escala de 0 a 1
        normalizador = MinMaxScaler(feature_range=(0,1))
        
        # Variável com a base com valores normalizados entre 0 e 1
        base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
        
        # Ficar na mesma escala de base_treinamento
        normalizador_previsao = MinMaxScaler(feature_range=(0,1))
        normalizador_previsao.fit_transform(base_treinamento[:,0:1])# ,0:1 = primeiro atributo. Aqui é a coluna que eu quero prever.
        
        anteriores = [300] ### Considerar
        
        for anterior in anteriores:
            previsores = [] #lista
            preco_real = [] #lista
            preco_real_desnormalizado=[]
            
            # Deve começar a tentar prever a partir da posição 90, pois os primeiros valores são para tentar prever. 1242 é o numero de registros
            for i in range(anterior, linhas): # append é para adicionar dado na lista
                previsores.append(base_treinamento_normalizada[i-anterior:i, 0:colunas])# i = 0,1,2,3... . vai até i, indice 0 a 2
                # preenche preço real
                preco_real.append(base_treinamento_normalizada[i, 0]) # i = 90 , preço das 90 datas anteriores, 1 = umidade relativa
                preco_real_desnormalizado.append(base_treinamento[i, 0])
            #transformar dados para tipo numpy, necessário para passar para rede neural
            previsores, preco_real = np.array(previsores), np.array(preco_real)
            
            
        predictions=model.predict(previsores)  # Realiza a predição
        
        predictions=normalizador_previsao.inverse_transform(predictions) #Desnormaliza
        #preco_real=normalizador_previsao.inverse_transform(preco_real) #Desnormaliza
        #print(predictions[:, 0])
        
        predictions.mean() #Calcula a média da predição
        preco_real.mean()  #Calcula a média do valor real
        
        #####Não usa rq aqui pois não tem como comparar 2016 com 2021. É uma simulação.
        #R quadrado
        #print("")
        rq = r2_score(preco_real_desnormalizado, predictions)
        print(f"R Quadrado = {rq}")
        
        #Root Mean Squared Error
        rmse = sqrt(mean_squared_error(preco_real_desnormalizado, predictions))
        print(f"RMSE = {rmse}")
        #Plota o grafico para comparar a predição    
        #name = '__RQ='+str(rq)
        
        #Salvar arquivo com "predictions" de 2021 
        #np.savetxt('Diana.Modelo=_'+mod+'_Dados_de_=_'+nom, coordenadas, fmt="%f", delimiter=",")
        #np.savetxt('Diana.2021'+nom, predictions, fmt="%f", delimiter=",")
        #
        #Se prever 2021 talvez comentar linha abaixo
        #plt.plot(preco_real_desnormalizado, color = 'red', label = 'Temperatura 2016')
        plt.plot(predictions, color = 'blue', label = 'Previsão Temperatura 2021')
        plt.title('Previsão Temperatura 2021')
        plt.xlabel('Timestamps(momentos)')
        plt.ylabel('Temperatura')
        plt.legend()
        #plt.savefig('Modelo=_'+mod+'_Dados_de_=_'+nom+'.RQ='+str(rq)+'.png')
        plt.savefig('Modelo=_'+mod+'_Dados_de_=_'+nom+'.RMSE='+str(rmse)+'.RQ='+str(rq)+'.png')
        plt.show()
        plt.close()

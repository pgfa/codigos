# -*- coding: utf-8 -*-
"""
Criado em 

@author: César
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nome = 'verao20112016.r2.csv'
base = pd.read_csv(nome)


np.random.seed(19680801)
N = 50
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

#Pegar dados para gerar gráficos
combinacoes = base.iloc[:,0:1].values # Coluna Todas as colunas
r2 = base.iloc[:,7:8].values #Pega somente a coluna 1
plt.plot(combinacoes, r2, color = 'blue')
#plt.scatter(combinacoes, r2, color="red")
plt.scatter(combinacoes, r2, s=area, c=colors, alpha=0.5)
#plt.title('Gráfico com o resultados de R² dos modelos criados')
#plt.xlabel('Combinação')
#plt.ylabel('R Quadrado')
#plt.ylim([0.8, 1])
#plt.grid(True)
 # Fixar tamanho do eixo Y
#plt.legend()
#plt.savefig(nome+'.png')
plt.show()
#plt.close()

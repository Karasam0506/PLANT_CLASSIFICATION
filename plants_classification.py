"""Modulo para Análise de Classificações de Plantas."""

# Definições & Traduções.

# Tipos de Solo
#
# loam - argiloso: Uma mistura equilibrada de areia, silte e argila,
# conhecida por ser fértil e excelente para a agricultura e jardinagem,
# pois retém água e nutrientes de forma eficaz, enquanto ainda oferece
# boa drenagem.
#
# sandy - arenoso: É caracterizado por partículas grandes, baixa
# retenção de água e nutrientes, e alta drenagem. É comum em áreas
# áridas ou litorâneas, sendo menos fértil para a agricultura, mas
# ideal para plantas que preferem solos bem drenados.
#
# clay - argiloso: É caracterizado por partículas extremamente finas e
# tem alta capacidade de retenção de água e nutrientes, mas geralmente
# apresenta drenagem lenta. É conhecida por formar uma textura densa e
# pegajosa quando molhada.

# Frequência de Irrigação
#
# bi-weekly: Irrigação realizada duas vezes por semana.
#
# weekly: Irrigação realizada uma vez por semana.
#
# daily: Irrigação realizada diariamente.

# Tipos de Fertilizante
#
# none: De acordo com os comentários respondidos pelo proprietário dos
# dados, o valor "none" na coluna "Fertilizer_Type" indica que não foram
# utilizados fertilizantes para o desenvolvimento da plantação.
# link: https://www.kaggle.com/datasets/gorororororo23/plant-growth-data-classification/discussion/520373
#
# chemical: Fertilizantes produzidos por meio de processos químicos,
# geralmente à base de compostos sintéticos que fornecem nutrientes
# essenciais para as plantas.
#
# organic: Fertilizantes de origem natural, criados a partir de matéria
# orgânica, como esterco, restos vegetais ou compostagem, utilizados para
# enriquecer o solo de maneira sustentável.


#%% Importação de Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from matplotlib.gridspec import GridSpec

#%% Importação dos dados
plantas_ori = pd.read_csv( "BASE\\plant_growth_data.csv")

#%% Exploração dos dados.
# Tamanho do arquivo
print( plantas_ori.shape )

# Apresentação inicial.
plantas_ori.head()

# Mínimos e Quartis
plantas_ori.describe()

# Análise de linhas nulas.
print(plantas_ori.isna().sum() )

plantas_ori.info()

# Distribuição de frequência das categorias.
plantas_ori.value_counts( "Water_Frequency", normalize=True)

plantas_ori.value_counts( "Fertilizer_Type", normalize=True)

plantas_ori.value_counts( "Soil_Type", normalize=True)

plantas_ori.value_counts( "Growth_Milestone", normalize=True)

print( plantas_ori.dtypes )

#%% Alteração do nome das Colunas.

new_columms = ["SOLO", "HORA_SOL", "FREQ_AGUA", "TIP_FTL", "TEMPE", "HUMID", "CRESC"]

plantas_ori.columns = new_columms

df_plantas = plantas_ori

col_numeric = ["HORA_SOL", "HUMID", "TEMPE", "CRESC"]

str_numeric = ["SOLO", "TIP_FTL", "FREQ_AGUA"]


#%% Análises descritivas.

# Análises das Horas de Sol.

## Descritivos

df_plantas["HORA_SOL"].describe()

## Histograma
sns.histplot( data = df_plantas, x = "HORA_SOL", bins = 20)

## Box-plot
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 3, figure=fig)  # Define uma grade 2x3

# Criando o gráfico na linha superior, ocupando toda a largura
ax1 = fig.add_subplot(gs[0, :])  # Linha 0, todas as colunas
sns.boxplot( data = df_plantas, x = "HORA_SOL", ax = ax1)

# Criando os 3 gráficos na linha inferior
ax2 = fig.add_subplot(gs[1, 0])  # Linha 1, coluna 0
sns.boxplot( data = df_plantas, x = "HORA_SOL", hue = "SOLO", ax = ax2)

ax3 = fig.add_subplot(gs[1, 1])  # Linha 1, coluna 1
sns.boxplot( data = df_plantas, x = "HORA_SOL", hue = "TIP_FTL", ax = ax3)

ax4 = fig.add_subplot(gs[1, 2])  # Linha 1, coluna 2
sns.boxplot( data = df_plantas, x = "HORA_SOL", hue = "FREQ_AGUA", ax = ax4)

# Criando os 3 gráficos na linha inferior
ax5 = fig.add_subplot(gs[2, 0])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "HORA_SOL", hue = "SOLO", ax = ax5)

ax6 = fig.add_subplot(gs[2, 1])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "HORA_SOL", hue = "TIP_FTL", ax = ax6)

ax7 = fig.add_subplot(gs[2, 2])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "HORA_SOL", hue = "FREQ_AGUA", ax = ax7)

# Análises da Temperatura.

## Descritivo
df_plantas["TEMPE"].describe()

## Histograma
sns.histplot( data = df_plantas, x = "TEMPE", bins = 20)

## Box-Plot
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 3, figure=fig)  # Define uma grade 2x3

# Criando o gráfico na linha superior, ocupando toda a largura
ax1 = fig.add_subplot(gs[0, :])  # Linha 0, todas as colunas
sns.boxplot( data = df_plantas, x = "TEMPE", ax = ax1)

# Criando os 3 gráficos na linha inferior
ax2 = fig.add_subplot(gs[1, 0])  # Linha 1, coluna 0
sns.boxplot( data = df_plantas, x = "TEMPE", hue = "SOLO", ax = ax2)

ax3 = fig.add_subplot(gs[1, 1])  # Linha 1, coluna 1
sns.boxplot( data = df_plantas, x = "TEMPE", hue = "TIP_FTL", ax = ax3)

ax4 = fig.add_subplot(gs[1, 2])  # Linha 1, coluna 2
sns.boxplot( data = df_plantas, x = "TEMPE", hue = "FREQ_AGUA", ax = ax4)

# Criando os 3 gráficos na linha inferior
ax5 = fig.add_subplot(gs[2, 0])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "TEMPE", hue = "SOLO", ax = ax5)

ax6 = fig.add_subplot(gs[2, 1])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "TEMPE", hue = "TIP_FTL", ax = ax6)

ax7 = fig.add_subplot(gs[2, 2])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "TEMPE", hue = "FREQ_AGUA", ax = ax7)

# Fazer Anova para a temperatura em fertilizantes e frequencia de hidratação

# Análises da Humidade.

## Descritivo
df_plantas["HUMID"].describe()

## Histograma
sns.histplot( data = df_plantas, x = "HUMID", bins = 15)

## Box-plot
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 3, figure=fig)  # Define uma grade 2x3

# Criando o gráfico na linha superior, ocupando toda a largura
ax1 = fig.add_subplot(gs[0, :])  # Linha 0, todas as colunas
sns.boxplot( data = df_plantas, x = "HUMID", ax = ax1)

# Criando os 3 gráficos na linha do meio
ax2 = fig.add_subplot(gs[1, 0])  # Linha 1, coluna 0
sns.boxplot( data = df_plantas, x = "HUMID", hue = "SOLO", ax = ax2)

ax3 = fig.add_subplot(gs[1, 1])  # Linha 1, coluna 1
sns.boxplot( data = df_plantas, x = "HUMID", hue = "TIP_FTL", ax = ax3)

ax4 = fig.add_subplot(gs[1, 2])  # Linha 1, coluna 2
sns.boxplot( data = df_plantas, x = "HUMID", hue = "FREQ_AGUA", ax = ax4)

# Criando os 3 gráficos na linha inferior
ax5 = fig.add_subplot(gs[2, 0])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "HUMID", hue = "SOLO", ax = ax5)

ax6 = fig.add_subplot(gs[2, 1])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "HUMID", hue = "TIP_FTL", ax = ax6)

ax7 = fig.add_subplot(gs[2, 2])  # Linha 1, coluna 0
sns.kdeplot( data = df_plantas, x = "HUMID", hue = "FREQ_AGUA", ax = ax7)

# Fazer ANOVA para humidade e Solo

#%%Correlações.

sns.pairplot( data = df_plantas[["TEMPE", "HUMID", "HORA_SOL"]])
# Em via dos gráficos de pontos entre as variáveis contínuas, é visível que
# não há relação.

# Correlação de Pearson
df_plantas = pd.get_dummies(df_plantas, columns=['SOLO'], prefix='SOLO', drop_first=False)
df_plantas = pd.get_dummies(df_plantas, columns=['FREQ_AGUA'], prefix='FREQ', drop_first=False)
df_plantas = pd.get_dummies(df_plantas, columns=['TIP_FTL'], prefix='FTL', drop_first=False)


col_corres = [ col for col in df_plantas.columns if col not in str_numeric]

corr = df_plantas[col_corres].corr()
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr,annot = True, cmap="YlOrBr")
# Aplicando a matriz de correlação utilizando a correlação de Pearson,
# é perceptível verificar que não há uma correlação linear forte entre as
# variáveis numéricas.

# Correlação de Spearman.
corr_spe = df_plantas[col_corres].corr(method='spearman')
mask_spe = np.triu(np.ones_like(corr_spe))
sns.heatmap(corr_spe,annot = True, cmap="YlOrBr", mask=mask_spe)

# Aplicando a matriz de correlação utilizando a correlação de Spearman,
# é perceptível verificar que não há uma correlação forte entre as
# variáveis numéricas.

# Análise de Crescimento
fig, ax = plt.subplots(3, 3, figsize=(20,10))

sns.histplot(data=df_plantas, x='TEMPE' , hue='CRESC', ax=ax[0,0])
sns.histplot(data=df_plantas, x='HORA_SOL', ax=ax[0,1], hue='CRESC')
sns.histplot(data=df_plantas, x='HUMID', ax=ax[0,2], hue='CRESC')

sns.boxplot(data=df_plantas, x='TEMPE', ax=ax[1,0], hue = "CRESC")
sns.boxplot(data=df_plantas, x='HORA_SOL', ax=ax[1,1], hue = "CRESC")
sns.boxplot(data=df_plantas, x='HUMID', ax=ax[1,2], hue = "CRESC")

sns.kdeplot( data = df_plantas, x = "TEMPE", hue = "CRESC",  ax=ax[2,0 ])
sns.kdeplot( data = df_plantas, x = "HORA_SOL", hue = "CRESC", ax=ax[2,1 ])
sns.kdeplot( data = df_plantas, x = "HUMID", hue = "CRESC", ax=ax[2,2 ])

# Fazer ANOVAs para a relação das variáveis quantitativas e crescimento.




























































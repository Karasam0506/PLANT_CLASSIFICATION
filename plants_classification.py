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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib.gridspec import GridSpec
import pingouin


#%% Definindo Funções

def verification( predito, test):
    """
    Avalia o desempenho de um modelo de classificação e exibe resultados relevantes.

    Esta função realiza as seguintes tarefas:
    1. Gera e exibe uma matriz de confusão utilizando o conjunto de valores 
    verdadeiros (`test`) e previstos (`predito`).
    2. Calcula métricas de desempenho do modelo, incluindo:
        - Acurácia (Accuracy): Proporção de predições corretas em relação ao total.
        - Sensibilidade (Recall): Proporção de casos positivos corretamente identificados.
        - Precisão (Precision): Proporção de predições positivas que são verdadeiramente corretas.

    Parâmetros:
    ----------
    predito : array-like
        Valores previstos pelo modelo de classificação.

    test : array-like
        Valores verdadeiros do conjunto de teste.

    Retornos:
    ---------
    None
        Os resultados são exibidos diretamente via `print` e a matriz de confusão é exibida em 
        um gráfico.
    """
    conf_matrix = confusion_matrix(test, predito)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap="viridis")

    acc = accuracy_score(predito, test)

    recall = recall_score(test, predito)

    precision = precision_score(test, predito, pos_label=1)

    print(f"""
    Resultados da Avaliação do Modelo:

    - Acurácia (Accuracy): {acc:.2f}

    - Sensibilidade (Recall): {recall:.2f}

    - Precisão (Precision): {precision:.2f}""")

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

#%% Alteração do nome das Colunas.

new_columms = ["SOLO", "HORA_SOL", "FREQ_AGUA", "TIP_FTL", "TEMPE", "HUMID", "CRESC"]

plantas_ori.columns = new_columms

df_plantas = plantas_ori

col_numeric = ["HORA_SOL", "HUMID", "TEMPE", "CRESC"]

str_numeric = ["SOLO", "TIP_FTL", "FREQ_AGUA"]


#%% Análises descritivas.

# Número de bins para o histograma
k = int(2*(len(df_plantas)**(1/3)))

# Análises das Horas de Sol.

## Descritivos

df_plantas["HORA_SOL"].describe()

## Histograma
sns.histplot( data = df_plantas, x = "HORA_SOL", bins = k)

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

anova_hora_solo = pingouin.anova( data = df_plantas,\
    dv = "HORA_SOL", between ="SOLO" )

print(anova_hora_solo)
anova_hora_ftl = pingouin.anova( data = df_plantas,\
    dv = "HORA_SOL", between ="TIP_FTL" )

print(anova_hora_ftl)

anova_hora_freq = pingouin.anova( data = df_plantas,\
    dv = "HORA_SOL", between ="FREQ_AGUA" )

print(anova_hora_freq)

# Anova entre as categorias do solo
pairwise_results = pingouin.pairwise_tests(data=df_plantas,
    dv="HORA_SOL",
    between="SOLO",
    padjust="bonf")

# Print pairwise_results
print(pairwise_results)


# Análises da Temperatura.

## Descritivo
df_plantas["TEMPE"].describe()

## Histograma
sns.histplot( data = df_plantas, x = "TEMPE", bins = k)

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
anova_tempe_freq = pingouin.anova( data = df_plantas,\
    dv = "TEMPE", between ="FREQ_AGUA" )

print(anova_tempe_freq)

anova_tempe_ftl = pingouin.anova( data = df_plantas,\
    dv = "TEMPE", between ="TIP_FTL" )

print(anova_tempe_ftl)

anova_tempe_solo = pingouin.anova( data = df_plantas,\
    dv = "TEMPE", between ="SOLO" )

print(anova_tempe_solo)
# Não há diferença entre a temperatura pela frequência de água.

# Análises da Humidade.

## Descritivo
df_plantas["HUMID"].describe()

## Histograma
sns.histplot( data = df_plantas, x = "HUMID", bins = k)

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

anova_humid_solo = pingouin.anova( data = df_plantas,\
    dv = "HUMID", between ="SOLO" )

print(anova_humid_solo)

anova_humid_ftl = pingouin.anova( data = df_plantas,\
    dv = "HUMID", between ="TIP_FTL" )

print(anova_humid_ftl)

anova_humid_freq = pingouin.anova( data = df_plantas,\
    dv = "HUMID", between ="FREQ_AGUA" )

print(anova_humid_freq)

#%%Correlações.

sns.pairplot( data = df_plantas[["TEMPE", "HUMID", "HORA_SOL"]])
# Em via dos gráficos de pontos entre as variáveis contínuas, é visível que
# não há relação.

# Correlação de Pearson
df_plantas = pd.get_dummies(df_plantas, columns=['SOLO'], prefix='SOLO')
df_plantas = pd.get_dummies(df_plantas, columns=['FREQ_AGUA'], prefix='FREQ')
df_plantas = pd.get_dummies(df_plantas, columns=['TIP_FTL'], prefix='FTL')

col_corres = [ col for col in df_plantas.columns if col not in str_numeric]

corr = df_plantas[col_corres].corr()
mask = np.triu(np.ones_like(corr))

plt.figure(figsize=(12, 8))

# Plotando o heatmap com melhorias
sns.heatmap(corr, annot=True, cmap="mako", mask = mask, annot_kws={"size": 10})

# Ajustando rotação dos rótulos
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Título e rótulos
plt.title("Matriz de Correlação de Spearman", fontsize=16)
plt.xlabel("Variáveis", fontsize=12)
plt.ylabel("Variáveis", fontsize=12)

plt.tight_layout()  # Ajuste automático dos elementos
plt.show()

# Aplicando a matriz de correlação utilizando a correlação de Pearson,
# é perceptível verificar que não há uma correlação linear forte entre as
# variáveis numéricas.

# Correlação de Spearman.
corr_spe = df_plantas[col_corres].corr(method='spearman')
mask_spe = np.triu(np.ones_like(corr_spe))

corr = df_plantas[col_corres].corr()
mask = np.triu(np.ones_like(corr))

plt.figure(figsize=(12, 8))

# Plotando o heatmap com melhorias
sns.heatmap(corr_spe, annot=True, cmap="mako", mask = mask_spe, annot_kws={"size": 10})

# Ajustando rotação dos rótulos
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Título e rótulos
plt.title("Matriz de Correlação de Spearman", fontsize=16)
plt.xlabel("Variáveis", fontsize=12)
plt.ylabel("Variáveis", fontsize=12)

plt.tight_layout()  # Ajuste automático dos elementos
plt.show()

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

anova_tempe_cresc = pingouin.anova( data = df_plantas,\
    dv = "TEMPE", between ="CRESC" )

print(anova_tempe_cresc)

anova_hora_cresc = pingouin.anova( data = df_plantas,\
    dv = "HORA_SOL", between ="CRESC" )

print(anova_hora_cresc)

anova_humid_cresc = pingouin.anova( data = df_plantas,\
    dv = "HUMID", between ="CRESC" )

print(anova_humid_cresc)

# Fazer ANOVAs para a relação das variáveis quantitativas e crescimento.

#%% Criação da Arvore de Descisão

# Separando Treinamento e teste

SEED = 46

df_resul_ori = df_plantas[["CRESC"]]

df_data_ori = df_plantas.drop("CRESC", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(df_data_ori, df_resul_ori, \
    test_size=0.3, random_state=SEED)

#Criação da primeira arvore

dt = DecisionTreeClassifier(max_depth=12, random_state=SEED, criterion='entropy')

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

verification( predito = y_pred, test = y_test)

print(classification_report(y_test, y_pred))

# Visualizar a árvore como gráfico
plt.figure(figsize=(20, 10))
plot_tree(dt,  filled=True, feature_names = X_train.columns)
plt.show()

#%% Utilizando o Baging

dt_one = DecisionTreeClassifier( random_state='SEED')
bc = BaggingClassifier(estimator=dt_one, n_estimators=500, random_state=SEED, oob_score=True)
bc.fit(X_train, y_train)

y_pred_bag = bc.predict(X_test)

verification(y_pred_bag, y_test)

print(classification_report(y_test, y_pred_bag))

#%% Random Forest

#Hiperparametros
params_rf = {"n_estimators":[ 8, 10 , 12 , 25, 50, 100,350,500],
    "max_features":["log2", "auto", "sqrt"],
    "min_samples_leaf":[2, 8,   10, 20, 30 ],
    "criterion": ["gini", "entropy", "log_loss" ],
    "max_depth": [None, 4, 8, 12, 20, 30]}


rf = RandomForestClassifier(random_state=SEED)

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                    param_grid=params_rf,
                    scoring='neg_mean_squared_error',
                    cv=3,
                    verbose=1,
                    n_jobs=-1)

grid_rf.fit( X_train, y_train )

best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred_forest = best_model.predict( X_test)

verification(predito = y_pred_forest, test = y_test )

print(classification_report(y_test, y_pred_forest))

#%% Análise de importância

feature_importances = best_model.feature_importances_

# Criar um DataFrame para visualização
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False).reset_index(drop = True)

importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()

#%% Gráfico de Linha.
# Configuração do tamanho do gráfico
plt.figure(figsize=(12, 6))

# Criar o gráfico de linha
plt.plot(importance_df['Feature'], importance_df['Importance'], marker='o', color='dodgerblue',
        linestyle='-', linewidth=2, markersize=8, label='Importância')

# Adicionar rótulos de valores diretamente nos pontos
for x, y in zip(importance_df['Feature'], importance_df['Importance']):
    plt.text(x, y, f'{y:.2f}', fontsize=9, ha='center', va='bottom', color='black')

# Configurar rótulos e título
plt.xlabel('Variáveis', fontsize=12)
plt.ylabel('Importância', fontsize=12)
plt.title('Importância das Variáveis', fontsize=14, weight='bold')

# Melhorar a legibilidade do eixo X
plt.xticks(rotation=45, ha='right', fontsize=10)

# Adicionar uma grade para melhorar a leitura
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar uma legenda
plt.legend(loc='upper right', fontsize=10)

# Ajustar layout
plt.tight_layout()

# Exibir o gráfico
plt.show()

#%% Gráfico de barras
# Criar o gráfico com Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df,
    palette='mako'
)

# Adicionar rótulos e título
plt.xlabel('Importância', fontsize=12)
plt.ylabel('Variáveis', fontsize=12)
plt.title('Importância das Variáveis', fontsize=14, weight='bold')

# Adicionar valores no final das barras
for index, value in enumerate(importance_df['Importance']):
    plt.text(value, index, f'{value:.2f}', va='center', fontsize=10)

plt.tight_layout()  # Ajustar o layout para evitar cortes
plt.show()

# Classificação do Crescimento de Plantas
O projeto a seguir apresenta uma análise de dados e o desenvolvimento de um modelo de classificação para identificar como fatores como tipo de solo, frequência de irrigação, uso de fertilizantes e outras variáveis influenciam o crescimento de uma determinada variedade de plantas.

## Dados
Este conjunto de dados "Classificação de Dados de Crescimento de Plantas" envolve uma tarefa de previsão que, normalmente, consiste em prever ou classificar o estágio de crescimento das plantas com base em fatores ambientais e de manejo fornecidos. Especificamente, o objetivo é prever o estágio ou marco de crescimento que uma planta atinge com base em variáveis como tipo de solo, horas de exposição à luz solar, frequência de rega, tipo de fertilizante, temperatura e umidade. Essa previsão pode ajudar a compreender como diferentes condições influenciam o crescimento das plantas e pode ser valiosa para otimizar práticas agrícolas ou o gerenciamento de estufas.

Descrição das colunas:
* Soil_Type (Tipo de Solo): O tipo ou composição do solo onde as plantas são cultivadas.
* Sunlight_Hours (Horas de Luz Solar): A duração ou intensidade da exposição à luz solar recebida pelas plantas.
* Water_Frequency (Frequência de Rega): A frequência com que as plantas são regadas, indicando o cronograma de irrigação.
* Fertilizer_Type (Tipo de Fertilizante): O tipo de fertilizante usado para nutrir as plantas.
* Temperature (Temperatura): As condições de temperatura ambiente em que as plantas são cultivadas.
* Humidity (Umidade): O nível de umidade no ambiente ao redor das plantas.
* Growth_Milestone (Marco de Crescimento): Descrições ou indicadores que representam estágios ou eventos significativos no processo de crescimento das plantas.

O Conjunto de dados está disponível em: https://www.kaggle.com/datasets/gorororororo23/plant-growth-data-classification/data

# Modelo
Para realizar a classificação e prever quais plantas têm maior probabilidade de crescer, utilizamos inicialmente um modelo de Árvore de Decisão. Esse modelo alcançou uma acurácia de 59%, indicando um desempenho moderado, porém com margem para melhorias.

Visando aprimorar os resultados, aplicamos o método de Floresta Aleatória (Random Forest), que combina múltiplas árvores de decisão para reduzir o overfitting e aumentar a generalização do modelo. No entanto, esse método resultou em uma acurácia de 57%, ligeiramente inferior ao modelo inicial.

Para otimizar ainda mais o desempenho, implementamos uma técnica de Bootstrap Aggregating (Bagging) em conjunto com a Floresta Aleatória. Essa abordagem consiste em criar várias subamostras do conjunto de dados original, treinar modelos independentes em cada subamostra e agregar suas previsões. Com isso, conseguimos elevar a acurácia para 60%.

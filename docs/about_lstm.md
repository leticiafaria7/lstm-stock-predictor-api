# Long Short-Term Memory (LSTM)

## ⏱️ Séries temporais

Séries temporais são sequências de observações coletadas ao longo do tempo em intervalos regulares ou irregulares.<br>Diferentemente de dados tabulares tradicionais, a ordem cronológica das observações é fundamental, pois valores passados podem influenciar diretamente os valores futuros.<br>Exemplos incluem preços de ações, temperatura, consumo de energia e indicadores econômicos.

### Conceitos básicos

Uma série temporal pode apresentar diferentes padrões, como:
- tendência (crescimento ou queda ao longo do tempo)
- sazonalidade (comportamentos que se repetem periodicamente)
- ciclos (oscilações de longo prazo sem periodicidade fixa) e ruído (variações aleatórias).

Identificar esses componentes é importante para escolher um modelo adequado e interpretar corretamente os resultados.

### Janela temporal (Sliding Window)

Modelos de aprendizado de máquina não recebem toda a série como entrada, mas sim uma sequência de observações anteriores denominada janela temporal (*lookback window*).<br>Por exemplo, utilizando uma janela de 60 dias, o modelo aprende a prever o preço do dia seguinte com base nos 60 preços anteriores.

### Pré-processamento

O pré-processamento normalmente envolve:
- tratamento de valores ausentes
- ordenação cronológica dos dados
- normalização ou padronização das variáveis e divisão entre conjuntos de treino
- validação e teste respeitando a ordem temporal.

Em séries temporais, os dados nunca devem ser embaralhados, pois isso causaria vazamento de informação (*data leakage*).

## 🧠 Redes neurais

Redes neurais artificiais são modelos inspirados no funcionamento dos neurônios biológicos.<br>Elas aprendem relações complexas entre entradas e saídas por meio do ajuste iterativo de parâmetros internos chamados pesos e vieses.<br>O treinamento consiste em resolver um problema de otimização, buscando os valores desses parâmetros que minimizam uma função de perda.

### Conceitos básicos

- **Backpropagation**: Este algoritmo calcula o gradiente da função de perda em relação aos pesos da rede utilizando a regra da cadeia do cálculo diferencial. Esses gradientes são utilizados por um algoritmo de otimização para atualizar os pesos, reduzindo progressivamente o erro cometido durante o treinamento.

- **Pesos**: Representam a intensidade da influência de uma conexão entre dois neurônios. Durante o treinamento, esses valores são ajustados para que a rede consiga aprender os padrões presentes nos dados.

- **Vieses (Bias)**: É um parâmetro adicional associado a cada neurônio que permite deslocar a função de ativação, aumentando a flexibilidade do modelo. Sem esse termo, a capacidade de representação da rede seria significativamente reduzida.

- **Nós e arestas**: Uma rede neural é composta por nós (neurônios artificiais) organizados em camadas e conectados por arestas. Cada aresta possui um peso associado, responsável por determinar a contribuição da saída de um neurônio para o próximo.

- **Funções de ativação**: Introduzem não linearidade ao modelo, permitindo que a rede aprenda relações complexas entre as variáveis. Sem elas, uma rede profunda seria equivalente a uma única transformação linear.
    - **ReLU (Rectified Linear Unit):** retorna o máximo entre 0 e a entrada. É a função mais utilizada em camadas ocultas devido à simplicidade computacional e à redução do problema de gradientes muito pequenos.
    - **Sigmoid:** mapeia os valores para o intervalo entre 0 e 1. É amplamente utilizada em problemas de classificação binária, especialmente na camada de saída.
    - **Tanh (Tangente Hiperbólica):** produz valores entre -1 e 1, mantendo média próxima de zero. É frequentemente utilizada em redes recorrentes, incluindo LSTM.

- **Função de perda (Loss Function)**: Mede o erro entre as previsões do modelo e os valores reais para uma determinada amostra ou lote de treinamento. O objetivo do treinamento é minimizar esse erro ao longo das épocas.

- **Função de custo (Cost Function)**: Representa uma agregação da função de perda sobre todo o conjunto de treinamento ou um lote de dados. Em muitos contextos, os termos são utilizados como sinônimos, embora tecnicamente a função de perda seja calculada por amostra e a função de custo seja uma média ou soma dessas perdas.

- **Gradiente**: Indica a direção e a intensidade da variação da função de custo em relação aos parâmetros do modelo. Algoritmos como Gradient Descent utilizam essa informação para atualizar os pesos na direção que reduz o erro.

- **Tensores**: São estruturas matemáticas que generalizam escalares, vetores e matrizes para múltiplas dimensões. Frameworks como TensorFlow e PyTorch representam todos os dados, pesos e operações utilizando tensores.

- **Regularização**: É o conjunto de técnicas utilizadas para reduzir o sobreajuste (*overfitting*), limitando a complexidade do modelo e melhorando sua capacidade de generalização. Entre as técnicas mais comuns estão L1, L2, Dropout e Early Stopping.

### Tipos de Redes Neurais

- **Feedforward Neural Network (FFNN)**
    - São redes neurais tradicionais em que a informação percorre apenas um sentido: da camada de entrada até a camada de saída.
    - São adequadas para problemas de classificação e regressão em dados tabulares, mas não modelam dependências temporais.

- **Convolutional Neural Network (CNN)**
    - Utilizam operações de convolução para extrair características locais dos dados, sendo amplamente empregadas em visão computacional e processamento de imagens.
    - Também podem ser adaptadas para séries temporais quando se deseja capturar padrões locais.

- **Recurrent Neural Network (RNN)**
    - Foram desenvolvidas para modelar dados sequenciais, mantendo um estado interno que transporta informações entre diferentes instantes da sequência.
    - Apesar dessa característica, apresentam dificuldades para aprender dependências de longo prazo devido ao problema do desaparecimento do gradiente.


## 📈 Long Short-Term Memory (LSTM)

É uma arquitetura especializada de Rede Neural Recorrente (RNN) desenvolvida para aprender dependências de longo prazo em dados sequenciais.<br>Seu mecanismo interno de memória permite preservar informações relevantes durante longos períodos, tornando-a amplamente utilizada em previsão de:
- Séries temporais financeiras
- Processamento de linguagem natural
- Reconhecimento de fala
- Sensores IoT
- Previsão meteorológica

### Conceitos básicos

- **Vanishing Gradient:** Este problema ocorre quando os gradientes se tornam progressivamente menores durante o treinamento, dificultando a atualização dos pesos das primeiras camadas ou dos primeiros instantes de uma sequência. Como consequência, uma RNN tradicional tende a "esquecer" informações distantes no tempo. A arquitetura LSTM foi projetada justamente para mitigar esse problema.

- **Arquitetura da LSTM:** A principal característica da LSTM é a existência de uma célula de memória (*Cell State*) controlada por portas (*gates*) que regulam quais informações devem ser descartadas, armazenadas ou utilizadas na saída da rede.

    - Forget Gate: decide quais informações armazenadas na memória devem ser descartadas. Esse mecanismo impede que informações irrelevantes continuem influenciando as previsões futuras.
    - Input Gate: determina quais novas informações serão adicionadas ao estado de memória da célula. Dessa forma, apenas informações consideradas relevantes são incorporadas ao conhecimento acumulado da rede.
    - Cell State: representa a memória de longo prazo da LSTM. Ele percorre toda a sequência transportando informações importantes e sendo atualizado pelas portas de entrada e esquecimento.
    - Output Gate: define quais informações presentes na memória serão utilizadas para produzir a saída do instante atual e atualizar o estado oculto da rede.

### Modelagem

- **Definição dos hiperparâmetros:** O desempenho de uma LSTM depende da escolha adequada de hiperparâmetros, como tamanho da janela temporal (*lookback*), número de camadas LSTM, quantidade de neurônios por camada, taxa de aprendizado (*learning rate*), tamanho do lote (*batch size*), número de épocas e taxa de *dropout*. Esses parâmetros normalmente são ajustados por experimentação ou técnicas de busca automática, como *Grid Search* e *Random Search*.

- **Avaliação:** Como a previsão de preços é um problema de regressão, o desempenho do modelo é avaliado utilizando métricas como:
    - MAE (*Mean Absolute Error*)
    - RMSE (*Root Mean Squared Error*)
    - MAPE (*Mean Absolute Percentage Error*)
    
    Além das métricas numéricas, é comum comparar visualmente as séries de valores reais e previstos para verificar se o modelo consegue acompanhar as tendências do mercado.

# Construção da Arquitetura LSTM em Python

## Visão Geral da Arquitetura LSTM

**Long Short-Term Memory (LSTM)** são um tipo de Rede Neural Recorrente (Recurrent Neural Network - RNN) para modelar dados sequenciais e resolver o problema do desaparecimento do gradiente (*vanishing gradient*), comum em RNNs tradicionais.

Diferentemente de redes neurais convencionais, que assumem que todas as amostras são independentes, as LSTMs são capazes de armazenar informações relevantes ao longo de uma sequência temporal por meio de um mecanismo interno de memória composto por portas (*gates*). Essas portas controlam quais informações devem ser esquecidas, quais devem ser armazenadas e quais serão utilizadas para produzir a saída da rede.

Essa característica torna as LSTMs especialmente adequadas para problemas como:

- Previsão de séries temporais;
- Processamento de linguagem natural;
- Reconhecimento de voz;
- Detecção de padrões em dados sequenciais;
- Predição de valores financeiros.

Na implementação deste projeto, a arquitetura foi construída utilizando a API **Keras**, presente no **TensorFlow**, permitindo uma definição modular e flexível da rede. A função `generate_lstm_model()` recebe como parâmetros todos os hiperparâmetros necessários para configurar a arquitetura dinamicamente, definidos a partir do algoritmo Optuna e específicos para cada ticker, possibilitando sua utilização durante processos de otimização.

## Construção da Arquitetura

A arquitetura é construída seguindo uma sequência lógica de etapas, descritas a seguir.

### 1. Seleção do Otimizador

O primeiro passo consiste em definir qual algoritmo será responsável pela atualização dos pesos da rede durante o treinamento.

```python
if optimizer_name == "Adam":
    optimizer = Adam(learning_rate)

elif optimizer_name == "Nadam":
    optimizer = Nadam(learning_rate)

else:
    optimizer = RMSprop(learning_rate)
```

O parâmetro `optimizer_name` permite selecionar dinamicamente entre três algoritmos de otimização:

- **Adam**
- **Nadam**
- **RMSprop**

Todos recebem como parâmetro a taxa de aprendizado (`learning_rate`), responsável por controlar o tamanho dos passos realizados durante o processo de otimização.

A escolha do otimizador influencia diretamente a velocidade de convergência e a estabilidade do treinamento.

### 2. Inicialização do Modelo Sequencial

Após definir o otimizador, é criado o modelo utilizando a API Sequencial do Keras.

```python
model = Sequential()
```

A classe `Sequential` organiza as camadas da rede em uma sequência linear, onde a saída de uma camada torna-se automaticamente a entrada da próxima.

### 3. Primeira Camada LSTM

A primeira camada recorrente é adicionada ao modelo.

```python
model.add(LSTM(
    units=units_1,
    return_sequences=True,
    recurrent_dropout=recurrent_dropout,
    kernel_regularizer=l2(l2_lambda),
    input_shape=(X_train.shape[1], X_train.shape[2])
))
```

Esta camada possui diversas configurações importantes.

#### Número de neurônios

```python
units=units_1
```

Define a quantidade de células LSTM responsáveis por extrair padrões temporais da sequência de entrada.

Quanto maior esse valor, maior será a capacidade de representação da rede, porém maior também será o custo computacional e o risco de sobreajuste (overfitting).

#### Formato da entrada

```python
input_shape=(X_train.shape[1], X_train.shape[2])
```

Define o formato esperado pela rede.

Nesse projeto:

- `X_train.shape[1]` representa o número de passos temporais (*timesteps*);
- `X_train.shape[2]` representa o número de variáveis (features) em cada instante.

Assim, cada amostra é composta por uma sequência temporal contendo múltiplas variáveis.

#### Retorno da sequência completa

```python
return_sequences=True
```

Esse parâmetro faz com que a camada produza uma saída para **cada instante da sequência**, e não apenas para o último.

Isso é necessário porque existe uma segunda camada LSTM logo em seguida, que utilizará toda a sequência produzida pela primeira camada como entrada.

#### Dropout recorrente

```python
recurrent_dropout=recurrent_dropout
```

O *recurrent dropout* aplica regularização apenas nas conexões recorrentes da LSTM.

Seu objetivo é reduzir o sobreajuste (*overfitting*) durante o treinamento.

#### Regularização L2

```python
kernel_regularizer=l2(l2_lambda)
```

Aplica penalização L2 aos pesos da camada.

Essa técnica reduz a magnitude dos pesos aprendidos, ajudando a controlar a complexidade do modelo e melhorando sua capacidade de generalização.

### 4. Primeira Camada de Dropout

Após a primeira LSTM, é aplicada uma camada de Dropout.

```python
model.add(Dropout(dropout))
```

Durante o treinamento, essa camada desativa aleatoriamente uma fração dos neurônios definida pelo parâmetro `dropout`.

Esse procedimento reduz a dependência entre neurônios e minimiza o risco de sobreajuste.

### 5. Segunda Camada LSTM

A segunda camada recorrente é adicionada logo após o Dropout.

```python
model.add(LSTM(
    units=units_2,
    recurrent_dropout=recurrent_dropout,
    kernel_regularizer=l2(l2_lambda)
))
```

Essa camada recebe como entrada toda a sequência produzida pela primeira LSTM.

Diferentemente da primeira camada, não utiliza:

```python
return_sequences=True
```

Como consequência, apenas o estado oculto correspondente ao último instante da sequência é utilizado como representação final da informação temporal.

Essa representação será utilizada pelas camadas densas seguintes.

### 6. Segunda Camada de Dropout

Após a segunda LSTM é aplicada outra camada de Dropout.

```python
model.add(Dropout(dropout))
```

Sua função permanece a mesma:

- reduzir overfitting;
- melhorar a capacidade de generalização da rede.

### 7. Camadas Densas Intermediárias

Após a extração das características temporais, o modelo pode incluir uma quantidade variável de camadas totalmente conectadas.

```python
for _ in range(n_dense):
    model.add(Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_lambda)
    ))
```

O número dessas camadas é definido pelo hiperparâmetro:

```python
n_dense
```

Cada camada possui:

- `dense_units` neurônios;
- função de ativação ReLU;
- regularização L2.

A utilização dessas camadas permite que o modelo aprenda relações não lineares entre as características extraídas pelas camadas LSTM antes da etapa final de predição.

### 8. Camada de Saída

A última camada da rede é responsável por produzir a previsão.

```python
model.add(Dense(1))
```

Ela contém apenas um neurônio, adequado para problemas de regressão, produzindo um único valor contínuo como saída.

Nenhuma função de ativação é especificada, fazendo com que seja utilizada uma ativação linear, apropriada para tarefas de previsão numérica.

### 9. Compilação do Modelo

Após a construção da arquitetura, o modelo é compilado.

```python
model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)
```

Durante essa etapa são definidos os componentes utilizados no treinamento.

#### Otimizador

```python
optimizer=optimizer
```

Utiliza o algoritmo selecionado na primeira etapa.

#### Função de perda

```python
loss="mse"
```

É utilizada a função **Mean Squared Error (MSE)**, amplamente empregada em problemas de regressão.

Essa função calcula a média dos quadrados das diferenças entre os valores previstos e os valores reais.

#### Métrica de avaliação

```python
metrics=["mae"]
```

Além da função de perda, é utilizada a métrica **Mean Absolute Error (MAE)**.

Enquanto o MSE é empregado para otimização do modelo, o MAE fornece uma medida intuitiva do erro médio absoluto entre previsões e valores observados.

### 10. Retorno do Modelo

Por fim, a função retorna a arquitetura completamente configurada.

```python
return model
```

Esse objeto pode então ser utilizado para treinamento utilizando o método `fit()`, avaliação com `evaluate()` ou geração de previsões através de `predict()`.

## Fluxo Geral da Arquitetura

A arquitetura construída pode ser resumida da seguinte forma:

```
Entrada
    │
    ▼
LSTM (units_1)
    │
    ▼
Dropout
    │
    ▼
LSTM (units_2)
    │
    ▼
Dropout
    │
    ▼
n camadas Dense (ReLU)
    │
    ▼
Dense (1)
    │
    ▼
Saída
```

Essa organização combina a capacidade das camadas LSTM de modelar dependências temporais de longo prazo com o poder das camadas densas para aprender relações não lineares, formando uma arquitetura adequada para problemas de previsão em séries temporais.
# Modelo LSTM | Predição de valor de fechamento na bolsa de valores

*Tech Challenge da Fase 2 da [pós-graduação em Engenharia de Machine Learning FIAP](https://postech.fiap.com.br/curso/machine-learning-engineering/)*

> 📈 Link para a API (em breve)

> 🎥 Vídeo com demonstração técnica do projeto (em breve)

## 🎯 Sobre o projeto
O projeto tem como objetivo a construção e deploy de um modelo de redes neurais Long-Short Term Memory (LSTM) para prever valor de fechamento das ações de uma empresa da bolsa.

- [Sobre LSTM](docs/about_lstm.md) (em construção)
- [Sobre as variáveis usadas no modelo](docs/about_stock_predicting_variables.md) (em construção)

## ⚙️ Funcionalidades
- **Pré-processamento dos dados:**
  - Coleta dos dados com as bibliotecas `yfinance` e `bcb`
  - Feature engineering
  - Armazenamento em banco
  - Orquestração com Airflow
- **Desenvolvimento do modelo LSTM:**
  - Construção
  - Treinamento
  - Avaliação
  - Exportação
  - Monitoramento do modelo
- **Deploy da API:**
  - Criação da API
  - Criação do dash
  - Docker
  - Documentação
  - Monitoramento da API (prometheus / grafana)

## 📐 Arquitetura
> Plano arquitetural (em breve)

## 📂 Estrutura do projeto
> ⚙️ *Em construção*
```
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── inference/
│
├── models/
│   └── lstm_model.keras
│
├── api/
│   ├── app.py
│   └── routes.py
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│
├── docker/
│
├── tests/
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🧭 Rotas da API (Endpoints)
Endpoint | Descrição
--- | ---
POST |
GET | 

## 📄 Documentação

## 💻 Instruções para execução

## 🚀 Evolução do projeto

Limitações: o mercado é influenciado por eventos externos: taxa de juros, inflação, eleições, guerras, resultados trimestrais, notícias

# Sprint 4: Detecção de Risco com RPA + IA

Este projeto implementa um pipeline de hiperautomação que coleta dados de produtos (RPA), os processa e utiliza um modelo de Machine Learning (Regressão Logística) para classificar produtos como "Legítimos" ou "Suspeitos", gerando um indicador de risco.

## Estrutura dos Arquivos

O projeto é dividido nos seguintes scripts, que devem ser executados em ordem:

-   `dataset_mercado_livre.csv`: (Pré-requisito) A base de dados bruta coletada pelo robô RPA (Scraping).
-   `dados_enriquecidos_com_alertas.csv`: (Pré-requisito) A base de dados enriquecida pelas heurísticas da Sprint 3.
-   `funcoes_analise.py`: Módulo de suporte que contém as funções de limpeza, enriquecimento e criação de features.
-   ---
-   **`0_gerar_base_treino.py`**: (IA - Passo 1) Script com rotulagem manual dos dados para criar a base de treino (`base_treino_manual.csv`).
-   **`1_treinar_modelo.py`**: (IA - Passo 2) Carrega a base de treino, treina o modelo de Regressão Logística e salva os artefatos (`modelo_risco.pkl`, `features_list.pkl`, etc.).
-   **`2_analise_sprint4.py`**: (RPA + IA - Passo 3) O pipeline principal. Carrega os dados "novos" (do `dataset_mercado_livre.csv`), aplica o modelo treinado (`.pkl`) e gera o relatório final (`relatorio_final_com_risco.csv`).
-   **`3_dashboard.py`**: (Relatório - Passo 4) Inicia um dashboard web interativo (Streamlit) para visualizar os resultados do relatório final.

## 🚀 Instruções de Execução

### 1. Pré-requisitos

-   Python 3.10 ou superior.
-   Os arquivos `dataset_mercado_livre.csv` e `dados_enriquecidos_com_alertas.csv` devem estar na mesma pasta.

### 2. Instalação das Dependências

Abra um terminal na pasta do projeto e instale as bibliotecas necessárias:

```bash
pip install pandas scikit-learn joblib streamlit altair
```

### 3. Execução do Pipeline (Passo a Passo)

Siga a ordem abaixo para executar o projeto do zero:

#### Passo 1: Gerar a Base de Treino (Simulada)

```bash
python 0_gerar_base_treino.py
```
* Saída: Cria o arquivo base_treino_manual.csv.

#### Passo 2: Treinar o Modelo de IA

```bash
python 1_treinar_modelo.py
```
* Saída: Cria os arquivos modelo_risco.pkl, avg_price_map.pkl, features_list.pkl e o gráfico grafico_matriz_confusao.png.

#### Passo 3: Executar o Pipeline de Análise (RPA + IA)

```bash
python 2_analise_sprint4.py
```
* Saída: Cria o relatório final relatorio_final_com_risco.csv.

#### Passo 4: Visualizar o Dashboard

```bash
streamlit run 3_dashboard.py
```
* Saída: Abrirá uma aba no seu navegador com o dashboard interativo.

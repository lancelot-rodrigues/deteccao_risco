# Módulo de Funções para Análise de Dados
# Contém funções para limpeza, enriquecimento, criação de features e visualização.

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

# Etapa 0: Configuração de Estilo para os Gráficos
def setup_visual_style():
    """Define um estilo visual padrão para todos os gráficos."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['figure.dpi'] = 100

# Etapa 1: Leitura e Limpeza dos Dados
def load_and_clean_data(filepath, separator=','):
    """Lê o arquivo CSV, renomeia colunas e faz a limpeza inicial."""
    print("Iniciando a leitura e limpeza dos dados...")
    
    try:
        df = pd.read_csv(filepath, sep=separator)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filepath}' não encontrado. Certifique-se de que ele está na mesma pasta que o script.")
        return None

    # Renomeia colunas para um padrão esperado
    colunas_para_renomear = {
        'nome_produto': 'titulo',
        'preco_produto': 'preco',
        'reviews_nota_media': 'avaliacao_nota',
        'reviews_quantidade_total': 'avaliacao_numero'
    }
    df.rename(columns=colunas_para_renomear, inplace=True)

    # Parser de preços no formato brasileiro (ex: "R$ 1.234,56")
    def parse_brazilian_price(price_str):
        if pd.isna(price_str): return np.nan
        try:
            s = str(price_str).replace('R$', '').strip()
            s = s.replace('.', '').replace(',', '.')
            return float(s)
        except (ValueError, TypeError): return np.nan

    if 'preco' in df.columns:
        df['preco'] = df['preco'].apply(parse_brazilian_price)
    
    # Conversão de colunas numéricas
    if 'avaliacao_nota' in df.columns:
        df['avaliacao_nota'] = pd.to_numeric(df['avaliacao_nota'].astype(str).str.replace(',', '.'), errors='coerce')
    
    if 'avaliacao_numero' in df.columns:
        df['avaliacao_numero'] = df['avaliacao_numero'].fillna(0)
        df['avaliacao_numero'] = df['avaliacao_numero'].astype(int)

    # Remoção de linhas com dados essenciais nulos
    df.dropna(subset=['preco', 'titulo'], inplace=True)
    
    print("Limpeza concluída. Resumo dos dados:")
    print(df.info())
    print("\nVerificando os dados limpos (5 primeiras linhas):")
    print(df[['titulo', 'preco', 'avaliacao_numero']].head())
    return df

# Etapa 2: Enriquecimento da Base de Dados
def enrich_data(df):
    """Cria novas colunas analíticas para aprofundar a análise."""
    print("\nIniciando o enriquecimento dos dados...")

    # Categorização de produtos baseada no título
    def categorize_product(title):
        title_lower = title.lower()
        if 'notebook' in title_lower or 'laptop' in title_lower:
            return 'Notebook'
        if 'impressora' in title_lower:
            return 'Impressora'
        return 'Suprimento de Impressão'
    df['categoria_produto'] = df['titulo'].apply(categorize_product)

    # Extração de atributos do título
    df['compatibilidade'] = np.where(df['titulo'].str.contains('compativel|compatível|gen[eé]rico|similar|tipo|remanufaturado', case=False, na=False, regex=True), 'Compatível', 'Original')
    df['capacidade'] = np.where(df['titulo'].str.contains('XL', case=False, na=False), 'XL (Alto Rendimento)', 'Padrão')
    df['modelo_cartucho'] = df['titulo'].str.extract(r'\b(662|664|667|954|122)\b', expand=False).fillna('Outro')
    
    # Extrair rendimento do título
    def extract_yield(text):
        if not isinstance(text, str): return np.nan
        match = re.search(r'(\d+)\s*(p[aá]ginas|pg|págs)\b', text, re.IGNORECASE)
        return int(match.group(1)) if match else np.nan
    
    df['rendimento_paginas'] = df['titulo'].apply(extract_yield)
    
    # Cálculo de custo por página
    df['custo_por_pagina'] = np.where(df['rendimento_paginas'] > 0, df['preco'] / df['rendimento_paginas'], np.nan)
    
    if df['custo_por_pagina'].notna().sum() > 0:
        print(f"Sucesso: Rendimento extraído do título para {df['custo_por_pagina'].notna().sum()} produtos.")
    else:
        print("Aviso: Não foi possível extrair o rendimento do título dos produtos.")

    print("Enriquecimento concluído.")
    return df

# Etapa 3: Criação de Features para ML
def create_features(df, avg_price_original_map):
    """Cria colunas numéricas (features) para o modelo de ML."""
    print("\nIniciando a criação de features para o ML...")

    # Tratamento de valores nulos para features numéricas
    df['preco'] = df['preco'].fillna(df['preco'].median())
    df['avaliacao_numero'] = df['avaliacao_numero'].fillna(0)
    df['custo_por_pagina'] = df['custo_por_pagina'].fillna(0)

    # Feature 1: Compatibilidade (Binário)
    df['feature_compativel'] = (df['compatibilidade'] == 'Compatível').astype(int)

    # Feature 2: Preço Anômalo (Binário)
    price_threshold = 0.5
    def check_price_anomaly(row):
        if row['compatibilidade'] == 'Original' and row['modelo_cartucho'] in avg_price_original_map:
            avg_price = avg_price_original_map[row['modelo_cartucho']]
            return 1 if row['preco'] < (avg_price * price_threshold) else 0
        return 0
    df['feature_preco_anomalo'] = df.apply(check_price_anomaly, axis=1)

    # Feature 3: Custo por Página Suspeito (Binário)
    cost_threshold = 0.01
    df['feature_custo_pagina_suspeito'] = ((df['custo_por_pagina'].notna()) & (df['custo_por_pagina'] < cost_threshold)).astype(int)

    # Feature 4: Baixa Reputação (Binário)
    review_threshold = 5
    df['feature_baixa_reputacao'] = (df['avaliacao_numero'] < review_threshold).astype(int)

    # Feature 5: Interação (Compatível + Baixa Reputação)
    df['feature_compativel_baixa_rep'] = ((df['feature_compativel'] == 1) & (df['feature_baixa_reputacao'] == 1)).astype(int)

    # Feature 6: Reputação do Vendedor (Binário)
    if 'reputacao_cor' in df.columns:
        bad_reputations = ['vermelho', 'laranja']
        df['feature_vendedor_ruim'] = df['reputacao_cor'].str.lower().isin(bad_reputations).astype(int)
    else:
        df['feature_vendedor_ruim'] = 0 # Valor neutro se a coluna não existir

    print("Criação de features concluída.")
    return df

# Etapa 4: Geração de Gráficos
def generate_visualizations(df):
    """Gera e salva os gráficos para a análise exploratória."""
    print("\nIniciando a geração das visualizações...")

    df_suprimentos = df[df['categoria_produto'] == 'Suprimento de Impressão'].copy()
    
    if df_suprimentos.empty:
        print("Nenhum 'Suprimento de Impressão' encontrado para gerar gráficos.")
        return

    # Filtrar outliers de preço para melhor visualização
    price_limit = df_suprimentos['preco'].quantile(0.95)
    df_filtered_price = df_suprimentos[df_suprimentos['preco'] <= price_limit]
    
    # Gráfico 1: Análise de Preços
    plt.figure()
    sns.boxplot(x='compatibilidade', y='preco', data=df_filtered_price, hue='compatibilidade', palette='viridis', order=['Original', 'Compatível'], legend=False)
    plt.title('Distribuição de Preços de Suprimentos')
    plt.xlabel('Tipo de Cartucho')
    plt.ylabel('Preço (R$)')
    plt.tight_layout()
    plt.savefig('grafico_1_preco_vs_compatibilidade.png')
    print("Gráfico 1 salvo.")
    plt.close()

    # Gráfico 2: Relação entre Preço e Avaliação
    plt.figure()
    sns.scatterplot(x='preco', y='avaliacao_nota', hue='compatibilidade', data=df_filtered_price.dropna(subset=['avaliacao_nota']), palette='magma', s=100, alpha=0.8)
    plt.title('Relação entre Preço e Nota de Avaliação')
    plt.xlabel('Preço (R$)')
    plt.ylabel('Nota Média de Avaliação')
    plt.legend(title='Compatibilidade')
    plt.tight_layout()
    plt.savefig('grafico_2_preco_vs_avaliacao.png')
    print("Gráfico 2 salvo.")
    plt.close()

    print("Visualizações geradas.")

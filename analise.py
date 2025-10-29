# --- SCRIPT: 2_analise_sprint4.py ---
# Executa o pipeline principal de análise (RPA + IA).

import pandas as pd
import joblib
import sys

# Importação de funções locais
try:
    from funcoes_analise import (
        load_and_clean_data, 
        enrich_data, 
        create_features, 
        setup_visual_style, 
        generate_visualizations
    )
except ImportError:
    print("ERRO: Arquivo 'funcoes_analise.py' não encontrado.")
    sys.exit()

print("--- Iniciando Pipeline de Análise e Detecção ---")

# 1. Carregamento do Modelo de IA
try:
    modelo_ia = joblib.load('modelo_risco.pkl')
    avg_price_map = joblib.load('avg_price_map.pkl')
    features_list = joblib.load('features_list.pkl')
    print("Modelo de IA e artefatos carregados com sucesso.")
except FileNotFoundError:
    print("ERRO: Arquivos 'modelo_risco.pkl', 'avg_price_map.pkl' ou 'features_list.pkl' não encontrados.")
    print("Por favor, execute o script '1_treinar_modelo.py' primeiro.")
    sys.exit()

# 2. Carregamento dos Dados (Simulação de RPA)
csv_filepath = 'dataset_mercado_livre.csv' 
print(f"Carregando dados 'novos' de: {csv_filepath}")

df_novos = load_and_clean_data(csv_filepath, separator=',')

if df_novos is None:
    print(f"ERRO: Falha ao carregar os dados de '{csv_filepath}'.")
    sys.exit()

# 3. Aplicação da Camada de IA
print("Aplicando enriquecimento e engenharia de features...")
df_enriquecido = enrich_data(df_novos.copy())
df_com_features = create_features(df_enriquecido, avg_price_map)

# Preparar dados para o modelo
X_para_prever = df_com_features[features_list].fillna(0)

print("\nAplicando modelo de IA para classificação e risco...")

# Aplicar modelo para classificação (0 ou 1)
df_final = df_com_features.copy()
df_final['classificacao_ia'] = modelo_ia.predict(X_para_prever)

# Aplicar modelo para probabilidade (Indicador de Risco)
probabilidades_risco = modelo_ia.predict_proba(X_para_prever)[:, 1]
df_final['indicador_de_risco_pct'] = (probabilidades_risco * 100).round(2)

print("Análise de risco concluída.")

# Mapear classificação para o relatório
mapa_risco = {0: 'Original/Legítimo', 1: 'Suspeito'}
df_final['classificacao_ia'] = df_final['classificacao_ia'].map(mapa_risco)

# 4. Geração de Relatórios
colunas_relatorio = [
    'titulo', 'preco', 'compatibilidade', 'modelo_cartucho', 
    'classificacao_ia', 'indicador_de_risco_pct'
]
colunas_existentes = [col for col in colunas_relatorio if col in df_final.columns]
df_relatorio_final = df_final[colunas_existentes].sort_values(by='indicador_de_risco_pct', ascending=False)

# 4a. Salvar Relatório em CSV
try:
    csv_filename = 'relatorio_final_com_risco.csv'
    df_relatorio_final.to_csv(csv_filename, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nRelatório final salvo com sucesso em: {csv_filename}")
except Exception as e:
    print(f"\nErro ao salvar o CSV final: {e}")

# 4b. Gerar Gráficos de Visualização
try:
    setup_visual_style()
    generate_visualizations(df_final) 
    print("Gráficos de visualização atualizados.")
except Exception as e:
    print(f"Erro ao gerar visualizações: {e}")

# 5. Exibição de Amostra
print("\n--- Amostra do Relatório de Risco (Maiores Riscos) ---")
print(df_relatorio_final.head(15).to_string())

print("\nPipeline RPA + IA concluído com sucesso!")

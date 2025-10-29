# --- SCRIPT: treinar_modelo.py ---
# Treinamento do modelo de classificação de risco.

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Importação de funções locais
try:
    from funcoes_analise import load_and_clean_data, enrich_data, create_features
except ImportError:
    print("ERRO: Arquivo 'funcoes_analise.py' não encontrado ou com erro.")
    print("Certifique-se de ter renomeado seu 'analise.py' e adicionado a função 'create_features'.")
    exit()

# Configuração de Arquivos
BASE_DE_TREINO = 'base_treino_manual.csv'
MODELO_SAIDA = 'modelo_risco.pkl'
MAPA_PRECOS_SAIDA = 'avg_price_map.pkl'
FEATURES_LIST_SAIDA = 'features_list.pkl'

# Execução do Treinamento
print("--- Iniciando Treinamento do Modelo de Risco ---")

# 1. Carregamento dos Dados
print(f"Carregando base de treino: {BASE_DE_TREINO}")
df_treino = load_and_clean_data(BASE_DE_TREINO, separator=';')

if df_treino is None:
    print(f"ERRO: Não foi possível carregar a base de treino '{BASE_DE_TREINO}'.")
    exit()

# 2. Enriquecimento de Dados
df_treino_enriquecido = enrich_data(df_treino.copy())

# Calcular e salvar mapa de preços médios
avg_price_map = df_treino_enriquecido[
    df_treino_enriquecido['compatibilidade'] == 'Original'
].groupby('modelo_cartucho')['preco'].mean().to_dict()

joblib.dump(avg_price_map, MAPA_PRECOS_SAIDA)
print(f"Mapa de preços médios salvo em: {MAPA_PRECOS_SAIDA}")

# 3. Criação de Features
df_treino_features = create_features(df_treino_enriquecido, avg_price_map)

# 4. Definição de Features (X) e Alvo (y)
FEATURES_LIST = [
    'preco',
    'avaliacao_numero',
    'custo_por_pagina',
    'feature_compativel',
    'feature_preco_anomalo',
    'feature_custo_pagina_suspeito',
    'feature_baixa_reputacao',
    'feature_compativel_baixa_rep',
    'feature_vendedor_ruim'
]

TARGET = 'label_risco_real' 

# Validar lista de features
FEATURES_LIST = [f for f in FEATURES_LIST if f in df_treino_features.columns]
print(f"\nFeatures que serão usadas no modelo: {FEATURES_LIST}")

# Preparar dataframe final
df_treino_final = df_treino_features.dropna(subset=[TARGET])

X = df_treino_final[FEATURES_LIST]
y = df_treino_final[TARGET]

# 5. Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"\nDados de treino: {len(X_train)} | Dados de teste: {len(X_test)}")
print(f"Distribuição do alvo no treino:\n{y_train.value_counts(normalize=True)}")

# 6. Criação do Pipeline de ML
pipeline_ml = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', random_state=42)) 
    # class_weight='balanced' para lidar com classes desbalanceadas
])

# 7. Treinamento do Modelo
print("\nTreinando o modelo de Regressão Logística...")
pipeline_ml.fit(X_train, y_train)

# 8. Avaliação do Modelo
print("\n--- AVALIAÇÃO DO MODELO (nos dados de teste) ---")
y_pred = pipeline_ml.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Classe 0 (Legítimo)', 'Classe 1 (Suspeito)']))

# 9. Geração da Matriz de Confusão
try:
    print("Gerando Matriz de Confusão...")
    ConfusionMatrixDisplay.from_estimator(pipeline_ml, X_test, y_test, cmap='Blues', display_labels=['Legítimo', 'Suspeito'])
    plt.title('Matriz de Confusão (Dados de Teste)')
    plt.savefig('grafico_matriz_confusao.png')
    print("Gráfico 'grafico_matriz_confusao.png' salvo.")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar Matriz de Confusão (ignorando): {e}")


# 10. Salvamento dos Artefatos
print("\nSalvando artefatos do modelo...")
joblib.dump(pipeline_ml, MODELO_SAIDA)
joblib.dump(FEATURES_LIST, FEATURES_LIST_SAIDA)

print("\n--- SUCESSO! ---")
print(f"Modelo salvo em: {MODELO_SAIDA}")
print(f"Lista de features salva em: {FEATURES_LIST_SAIDA}")
print("Treinamento concluído.")

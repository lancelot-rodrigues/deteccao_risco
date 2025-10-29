import pandas as pd
import numpy as np

# Configuração
INPUT_FILE = 'dados_enriquecidos_com_alertas.csv'
OUTPUT_FILE = 'base_treino_manual.csv'

def simular_rotulagem_humana(row):
    """
    Simula a rotulagem de dados para criar o alvo (target) do modelo de ML.
    """
    
    # Regra 1: "Compatíveis" são tratados como legítimos (não-piratas).
    if row['compatibilidade'] == 'Compatível':
        return 0 # 0 = Legítimo

    # Regra 2: "Original" com alerta de suspeita.
    if row['compatibilidade'] == 'Original' and row['alerta_suspeita'] == True:
        return 1 # 1 = Suspeito

    # Regra 3: "Original" sem alerta (requer análise extra).
    if row['compatibilidade'] == 'Original' and row['alerta_suspeita'] == False:
        
        # Pista 3a: Reputação do vendedor ruim.
        if 'reputacao_cor' in row.index and pd.notna(row['reputacao_cor']):
            if row['reputacao_cor'].lower() in ['vermelho', 'laranja']:
                return 1 # Suspeito (vendedor ruim)

        # Pista 3b: Número de avaliações muito baixo.
        if pd.notna(row['avaliacao_numero']) and row['avaliacao_numero'] < 3:
            return 1 # Suspeito (baixas avaliações)

        # Caso padrão para Originais sem alerta
        return 0

    # Default: Legítimo
    return 0

# Execução Principal
print(f"Iniciando a geração da base de treino...")
print(f"Carregando arquivo de entrada: {INPUT_FILE}")

# Carregamento dos Dados
try:
    df = pd.read_csv(INPUT_FILE, sep=';')
except FileNotFoundError:
    print(f"ERRO: Arquivo '{INPUT_FILE}' não encontrado.")
    print("Certifique-se de que ele está na mesma pasta que este script.")
    exit()
except Exception as e:
    print(f"Erro ao ler o CSV: {e}")
    exit()

print("Aplicando lógica de rotulagem simulada em todos os produtos...")

# Aplicação da Rotulagem
df['label_risco_real'] = df.apply(simular_rotulagem_humana, axis=1)

# Conversão de Tipo
df['label_risco_real'] = df['label_risco_real'].astype(int)

print("\nDistribuição dos rótulos gerados:")
print(df['label_risco_real'].value_counts())

# Salvamento do Resultado
try:
    df.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nSUCESSO! Base de treino gerada e salva como: {OUTPUT_FILE}")
    print(f"Total de {len(df)} produtos rotulados.")
except Exception as e:
    print(f"\nERRO ao salvar o arquivo final: {e}")

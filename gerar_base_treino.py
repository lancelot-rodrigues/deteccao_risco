import pandas as pd
import numpy as np

# --- CONFIGURAÇÃO ---
INPUT_FILE = 'dados_enriquecidos_com_alertas.csv'
OUTPUT_FILE = 'base_treino_manual.csv'
# ---

def simular_rotulagem_humana(row):
    """
    Esta função simula um analista humano rotulando os dados
    para criar o alvo (target) do nosso modelo de ML.
    """
    
    # Regra 1: "Compatíveis" são tratados como legítimos (não-piratas).
    # O foco da "pirataria" é um produto que se *finge* de original.
    if row['compatibilidade'] == 'Compatível':
        return 0 # 0 = Legítimo

    # Regra 2: Se é "Original" e suas regras JÁ sinalizaram (alerta_suspeita == True).
    # Este é um suspeito óbvio.
    if row['compatibilidade'] == 'Original' and row['alerta_suspeita'] == True:
        return 1 # 1 = Suspeito

    # Regra 3: Se é "Original" e suas regras NÃO sinalizaram (alerta_suspeita == False).
    # Aqui simulamos a "inteligência humana" buscando mais pistas.
    if row['compatibilidade'] == 'Original' and row['alerta_suspeita'] == False:
        
        # Pista 3a: Reputação do vendedor (se existir e for ruim)
        if 'reputacao_cor' in row.index and pd.notna(row['reputacao_cor']):
            if row['reputacao_cor'].lower() in ['vermelho', 'laranja']:
                return 1 # O vendedor é ruim, mesmo que o preço pareça OK. Suspeito.

        # Pista 3b: Pouquíssimas avaliações, mesmo sendo "original".
        if pd.notna(row['avaliacao_numero']) and row['avaliacao_numero'] < 3:
            return 1 # Um produto original "famoso" com 2 avaliações? Suspeito.

        # Se passou por tudo isso, é provavelemente legítimo.
        return 0

    # Default para qualquer caso não pego
    return 0

# --- EXECUÇÃO PRINCIPAL ---
print(f"Iniciando a geração da base de treino...")
print(f"Carregando arquivo de entrada: {INPUT_FILE}")

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

# Aplicar a função para criar a nova coluna 'label_risco_real'
df['label_risco_real'] = df.apply(simular_rotulagem_humana, axis=1)

# Converter para número inteiro (importante para o ML)
df['label_risco_real'] = df['label_risco_real'].astype(int)

print("\nDistribuição dos rótulos gerados:")
print(df['label_risco_real'].value_counts())

# Salvar o novo arquivo CSV completo
try:
    df.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nSUCESSO! Base de treino gerada e salva como: {OUTPUT_FILE}")
    print(f"Total de {len(df)} produtos rotulados.")
except Exception as e:
    print(f"\nERRO ao salvar o arquivo final: {e}")
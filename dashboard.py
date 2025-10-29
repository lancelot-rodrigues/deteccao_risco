# --- SCRIPT: 3_dashboard.py ---
# (Rode com: streamlit run 3_dashboard.py)

import streamlit as st
import pandas as pd
import altair as alt # Para gráficos interativos

# --- 1. Configuração da Página ---
# set_page_config deve ser o primeiro comando streamlit
st.set_page_config(
    layout="wide", 
    page_title="Dashboard de Risco",
    page_icon="🚨"
)

# --- 2. Título e Introdução ---
st.title(" Dashboard de Detecção de Risco")
st.write(
    "Este dashboard interativo apresenta os resultados da análise de IA (RPA + Regressão Logística) "
    "sobre os produtos coletados. Use os filtros na barra lateral para explorar os dados."
)
st.divider()

# --- 3. Carregamento dos Dados ---
# @st.cache_data armazena os dados em cache para performance
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, sep=';')
        return df
    except FileNotFoundError:
        st.error(f"Erro: Arquivo '{filepath}' não encontrado.")
        st.info("Por favor, execute o script '2_analise_sprint4.py' primeiro para gerar o relatório.")
        return pd.DataFrame() # Retorna um dataframe vazio

DATA_FILE = 'relatorio_final_com_risco.csv'
df = load_data(DATA_FILE)

if not df.empty:

    # --- 4. Barra Lateral com Filtros ---
    st.sidebar.header("Painel de Filtros")

    # Filtro por Classificação da IA
    classificacao_opcoes = ['Todos'] + list(df['classificacao_ia'].unique())
    classificacao_filtro = st.sidebar.selectbox(
        "Filtrar por Classificação da IA:",
        options=classificacao_opcoes,
        index=0 # Default é "Todos"
    )

    # Filtro por Score de Risco (Slider)
    risco_slider = st.sidebar.slider(
        "Filtrar por Indicador de Risco (%):",
        min_value=0,
        max_value=100,
        value=(50, 100) # Default: mostrar produtos com 50% a 100% de risco
    )
    
    # --- 5. Aplicar Filtros ---
    df_filtrado = df.copy()
    
    # Aplicar filtro de classificação (se não for "Todos")
    if classificacao_filtro != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['classificacao_ia'] == classificacao_filtro]
    
    # Aplicar filtro de risco
    df_filtrado = df_filtrado[
        (df_filtrado['indicador_de_risco_pct'] >= risco_slider[0]) &
        (df_filtrado['indicador_de_risco_pct'] <= risco_slider[1])
    ]

    # --- 6. Exibir Métricas Principais (KPIs) ---
    st.subheader("Métricas Gerais da Análise")
    
    total_produtos = len(df)
    total_suspeitos = len(df[df['classificacao_ia'] == 'Suspeito'])
    risco_medio_total = df['indicador_de_risco_pct'].mean()
    
    # Métricas do DADO FILTRADO
    total_filtrado = len(df_filtrado)
    risco_medio_filtrado = df_filtrado['indicador_de_risco_pct'].mean() if total_filtrado > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Produtos (Geral)", f"{total_produtos}")
    col2.metric("Total 'Suspeitos' (Geral)", f"{total_suspeitos}")
    col3.metric("Risco Médio (Geral)", f"{risco_medio_total:.1f}%")
    col4.metric(
        label="Produtos no Filtro Atual", 
        value=f"{total_filtrado}", 
        delta=f"{total_filtrado - total_produtos} do total",
        delta_color="off"
    )
    
    st.divider()

# --- 7. Visualização de Dados (Gráficos) ---
    st.subheader("Visualização das Distribuições de Risco")

    # Definir a paleta de cores que usaremos
    color_scale = alt.Scale(domain=['Original/Legítimo', 'Suspeito'], range=['#4CAF50', '#FF5733'])

    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.write("#### 1. Contagem de Produtos por Faixa de Risco")
        # Gráfico de Barras Empilhadas (Contagem)
        # Mostra a *quantidade* de produtos em cada faixa de risco.
        
        binned_risk_axis = alt.X('indicador_de_risco_pct', bin=alt.Bin(maxbins=20), title='Faixa de Risco (%)')
        
        stacked_bar = alt.Chart(df_filtrado).mark_bar().encode(
            x=binned_risk_axis,
            
            # --- MUDANÇA PRINCIPAL AQUI ---
            # Removemos 'stack="normalize"' do eixo Y.
            # Agora, o eixo Y mostrará a CONTAGEM real de produtos,
            # e as barras terão alturas diferentes.
            y=alt.Y('count()', title='Contagem de Produtos'), 
            
            color=alt.Color('classificacao_ia', title='Classificação', scale=color_scale),
            
            tooltip=[
                alt.Tooltip('indicador_de_risco_pct', bin=alt.Bin(maxbins=20), title='Faixa de Risco (%)'),
                'classificacao_ia',
                'count()'
            ]

        ).properties(
            title='Distribuição de Produtos por Risco (Contagem)'
        ).interactive()
        st.altair_chart(stacked_bar, use_container_width=True)

    with col_graf2:
        st.write("#### 2. Densidade das Populações por Risco")
        # Gráfico de Densidade (KDE)
        # Mostra as duas "populações" de produtos e onde elas se concentram
        
        density_plot = alt.Chart(df_filtrado).transform_density(
            'indicador_de_risco_pct',
            as_=['risco', 'densidade'],
            groupby=['classificacao_ia']
        ).mark_area(opacity=0.7).encode(
            # --- CORREÇÃO AQUI ---
            # Precisamos ser explícitos sobre o tipo de dado ('Q' para quantitativo)
            # que a transformação 'transform_density' cria.
            x=alt.X('risco:Q', title='Indicador de Risco (%)'),
            y=alt.Y('densidade:Q', title='Densidade', axis=None), # :Q já estava aqui
            
            color=alt.Color('classificacao_ia', title='Classificação', scale=color_scale),
            
            tooltip=[
                alt.Tooltip('risco:Q', title='Indicador de Risco (%)'), # --- E AQUI ---
                'classificacao_ia'
            ]
        ).properties(
            title='Concentração de Produtos por Risco'
        ).interactive()
        st.altair_chart(density_plot, use_container_width=True)


    # Gráfico 3: Preço vs. Score de Risco (mantemos este, pois é útil)
    st.write("---") # Separador
    st.write("#### 3. Preço dos Produtos vs. Indicador de Risco")
    
    scatter_preco_risco = alt.Chart(df_filtrado).mark_circle(size=80, opacity=0.8).encode(
        # Vamos manter a escala de Preço linear, mas formatar o eixo
        x=alt.X('preco', title='Preço (R$)', axis=alt.Axis(format='~s')), 
        y=alt.Y('indicador_de_risco_pct', title='Indicador de Risco (%)'),
        color=alt.Color('classificacao_ia', title='Classificação', scale=color_scale),
        tooltip=['titulo', 'preco', 'indicador_de_risco_pct', 'classificacao_ia']
    ).properties(
        title='Preço dos Produtos vs. Indicador de Risco'
    ).interactive()
    st.altair_chart(scatter_preco_risco, use_container_width=True)


    st.divider() # Final do separador
    
    # --- 8. Tabela de Dados Detalhada ---
    # (O resto do seu código da tabela continua aqui, sem alterações)

    # --- 8. Tabela de Dados Detalhada ---
    st.subheader(f"Relatório Detalhado ({total_filtrado} produtos)")
    
    # Formatação visual do risco na tabela
    st.dataframe(
        df_filtrado.style.background_gradient(
            cmap='Reds', 
            subset=['indicador_de_risco_pct'],
            vmin=0,
            vmax=100
        ), 
        use_container_width=True,
        height=500
    )

else:
    st.warning("O arquivo de dados não foi carregado. Execute o script '2_analise_sprint4.py' e recarregue esta página.")
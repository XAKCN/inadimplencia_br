# =============================================================================
# EDA: INADIMPLENCIA BANCARIA BRASILEIRA - DADOS REAIS BACEN / IBGE
#
# Fontes (todas publicas e verificaveis):
#   - BACEN Olinda: taxas de juros e inadimplencia por modalidade/instituicao
#   - BACEN SGS: inadimplencia agregada, selic, IPCA, concessoes, saldo de credito
#   - IBGE SIDRA: desocupacao (PNAD), IPCA mensal, PIB per capita
#
# Saida (tudo em outputs/):
#   painel_eda_inadimplencia.png  => analise por modalidade/segmento (Olinda)
#   painel_eda_macro.png          => painel macroeconomico (SGS + IBGE)
#   imagens individuais de cada grafico
# =============================================================================

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SGS, PERIODO, SAIDA
from src.constants import MESES_PT
from src.pipeline_utils import configurar_exibicao, montar_painel_sgs
from src.visualization_utils import PainelSaida

from src.data_loaders import (
    carregar_serie_bacen, carregar_desemprego_ibge,
    carregar_pib_ibge, carregar_ipca_ibge, carregar_olinda_credito,
)
from src.exploratory_utils import (
    limpar_dados_inadimplencia_br, analisar_outliers_br,
    gerar_estatisticas_descritivas_br,
    imprimir_visao_geral_macro, gerar_estatisticas_descritivas_macro,
    analisar_dados_faltantes_macro, analisar_outliers_macro,
)
from src.plots_analise import (
    gerar_boxplots_outliers,
    gerar_distribuicoes_numericas, gerar_distribuicoes_categoricas,
    gerar_graficos_variavel_alvo, gerar_analises_bivariadas,
    gerar_analise_temporal, gerar_matriz_correlacao,
    gerar_analise_avancada, insights_finais_inadimplencia,
    # novos graficos Olinda
    gerar_ranking_instituicoes,
    gerar_bubble_chart_modalidades,
    gerar_evolucao_top_modalidades,
    gerar_violin_por_segmento,
    gerar_heatmap_instituicao_modalidade,
    gerar_concentracao_mercado,
    gerar_quartis_modalidade_temporal,
    gerar_juros_por_segmento,
)
from src.plots_macro import (
    gerar_graficos_contexto_macro,
    gerar_boxplots_outliers_macro, gerar_distribuicoes_numericas_macro,
    gerar_distribuicoes_temporais, gerar_series_temporais_completas,
    gerar_distribuicao_inadimplencia, gerar_bivariada_indicadores_macro,
    gerar_inadimplencia_por_periodo, gerar_matriz_correlacao_macro,
    gerar_analises_avancadas_macro, insights_finais_macro,
    # novos graficos macro
    gerar_pf_vs_pj,
    gerar_variacao_yoy,
    gerar_carteira_vs_inadimplencia,
    gerar_scatter_matrix_macro,
    gerar_analise_regimes,
    gerar_decomposicao_stl,
    gerar_acf_pacf,
    gerar_coeficientes_ols,
)

# ---------------------------------------------------------------------------
# Configuracao inicial
# ---------------------------------------------------------------------------
logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

configurar_exibicao(float_fmt='{:.3f}')

base_dir   = Path(__file__).resolve().parent
data_dir   = base_dir / 'data'
output_dir = base_dir / 'outputs'
data_dir.mkdir(exist_ok=True)

painel_macro = PainelSaida(
    base_dir=output_dir,
    arquivo_painel='painel_eda_macro.png',
    titulo_painel='EDA - Painel Macroeconomico de Inadimplencia (BACEN SGS + IBGE)',
    gerar_individuais=True,
)
painel_eda = PainelSaida(
    base_dir=output_dir,
    arquivo_painel='painel_eda_inadimplencia.png',
    titulo_painel='EDA - Inadimplencia por Modalidade e Segmento (BACEN Olinda)',
    gerar_individuais=True,
)

# ---------------------------------------------------------------------------
# 1. COLETA DE DADOS
# ---------------------------------------------------------------------------
logger.info("=" * 80)
logger.info("EDA - INADIMPLENCIA BANCARIA BRASILEIRA (BACEN / IBGE)")
logger.info("=" * 80)
logger.info("\n[1. COLETA DE DADOS - APIS PUBLICAS]")
logger.info("-" * 50)

logger.info("\n  >> BACEN SGS:")
df_inadt  = carregar_serie_bacen(SGS['inadimplencia_total'], PERIODO['data_inicial'], PERIODO['data_final'], 'inadimplencia_total', data_dir / 'serie_inadimplencia_total_bacen_2015_2025.csv')
df_inadpf = carregar_serie_bacen(SGS['inadimplencia_pf'],   PERIODO['data_inicial'], PERIODO['data_final'], 'inadimplencia_pf',    data_dir / 'serie_inadimplencia_pf_bacen_2018_2025.csv')
df_inadpj = carregar_serie_bacen(SGS['inadimplencia_pj'],   PERIODO['data_inicial'], PERIODO['data_final'], 'inadimplencia_pj',    data_dir / 'serie_inadimplencia_pj.csv')
df_selic  = carregar_serie_bacen(SGS['selic_aa'],           PERIODO['data_inicial'], PERIODO['data_final'], 'selic_aa',            data_dir / 'serie_selic.csv')
df_ipca   = carregar_serie_bacen(SGS['ipca_acum12m'],       PERIODO['data_inicial'], PERIODO['data_final'], 'ipca_acum12m',        data_dir / 'serie_ipca.csv')
df_conc   = carregar_serie_bacen(SGS['concessoes_bi'],      PERIODO['data_inicial'], PERIODO['data_final'], 'concessoes_bi',       data_dir / 'serie_concessoes.csv')
df_saldo  = carregar_serie_bacen(SGS['saldo_credito_bi'],   PERIODO['data_inicial'], PERIODO['data_final'], 'saldo_credito_bi',    data_dir / 'serie_saldo_credito.csv')

logger.info("\n  >> IBGE SIDRA:")
df_desemprego = carregar_desemprego_ibge(PERIODO['data_inicial_iso'], PERIODO['data_final_iso'], data_dir / 'serie_desemprego_ibge.csv', mensal_interpolado=True)
df_pib        = carregar_pib_ibge(PERIODO['ano_inicial'], PERIODO['ano_final'], data_dir / 'serie_pib_per_capita_ibge.csv')
df_ipca_ibge  = carregar_ipca_ibge(PERIODO['data_inicial_iso'], PERIODO['data_final_iso'], data_dir / 'serie_ipca_mensal_ibge.csv')

logger.info("\n  >> BACEN Olinda - Credito por Modalidade/Instituicao:")
df_olinda = carregar_olinda_credito(data_dir / 'taxas_juros_modalidade_bacen.csv',
                                     top=SAIDA['olinda_top'])

# ---------------------------------------------------------------------------
# 2. MONTAGEM DO PAINEL MACROECONOMICO (SGS + IBGE)
# ---------------------------------------------------------------------------
logger.info("\n  >> Montando painel macroeconomico integrado...")
df_macro = montar_painel_sgs(
    [df_inadt, df_inadpf, df_inadpj, df_selic, df_ipca, df_conc, df_saldo],
    series_ibge=[df_desemprego, df_ipca_ibge],
)

if not df_pib.empty:
    df_pib['ano_join']   = df_pib['data'].dt.year
    df_macro['ano_join'] = df_macro['data'].dt.year
    df_macro = pd.merge(df_macro, df_pib[['ano_join', 'pib_per_capita']], on='ano_join', how='left')
    df_macro.drop(columns='ano_join', inplace=True)

df_macro.sort_values('data', inplace=True)
df_macro.reset_index(drop=True, inplace=True)

df_macro['ano']       = df_macro['data'].dt.year
df_macro['mes']       = df_macro['data'].dt.month
df_macro['trimestre'] = df_macro['data'].dt.quarter
df_macro['mes_nome']  = df_macro['data'].dt.month.map(
    lambda m: MESES_PT[int(m) - 1] if pd.notna(m) else pd.NA
)

if 'inadimplencia_total' in df_macro.columns:
    s = df_macro['inadimplencia_total']
    df_macro['inadt_var_pp']  = s.diff()
    df_macro['inadt_var_pct'] = s.pct_change() * 100
    df_macro['inadt_mm3']     = s.rolling(3,  min_periods=1).mean()
    df_macro['inadt_mm6']     = s.rolling(6,  min_periods=1).mean()
    df_macro['inadt_mm12']    = s.rolling(12, min_periods=1).mean()
    df_macro['inadt_lag1']    = s.shift(1)
    df_macro['inadt_lag3']    = s.shift(3)
    df_macro['inadt_lag6']    = s.shift(6)
    df_macro['inadt_lag12']   = s.shift(12)

COLS_BASE = [c for c in ['inadimplencia_total', 'inadimplencia_pf', 'inadimplencia_pj',
                          'selic_aa', 'ipca_acum12m', 'ipca_mensal',
                          'concessoes_bi', 'saldo_credito_bi',
                          'desemprego', 'pib_per_capita'] if c in df_macro.columns]
COL_ALVO_MACRO = ('inadimplencia_total' if 'inadimplencia_total' in df_macro.columns
                   else 'inadimplencia_pf' if 'inadimplencia_pf' in df_macro.columns
                   else None)

logger.info(f"  Painel macro: {df_macro.shape[0]} obs x {df_macro.shape[1]} colunas")
logger.info(f"  Periodo     : {df_macro['data'].min().date()} -> {df_macro['data'].max().date()}")

# ---------------------------------------------------------------------------
# 3. PREPARACAO DOS DADOS OLINDA
# ---------------------------------------------------------------------------
if not df_olinda.empty:
    df_olinda['Mes'] = pd.to_datetime(df_olinda['Mes'], errors='coerce')
    for col in ['TaxaJurosAoMes', 'TaxaJurosAoAno', 'TaxaInadimplencia',
                'NumeroDeContratos', 'BaseDeCalculo']:
        if col in df_olinda.columns:
            df_olinda[col] = pd.to_numeric(df_olinda[col], errors='coerce')
    usar_olinda = True
    logger.info(f"\n  Olinda: {df_olinda.shape[0]:,} linhas x {df_olinda.shape[1]} colunas")
else:
    usar_olinda = False
    logger.warning("  [AVISO] Dataset Olinda indisponivel - painel EDA sera omitido.")

# ===========================================================================
# PAINEL MACRO  =>  outputs/painel_eda_macro.png
# ===========================================================================
logger.info("\n" + "=" * 80)
logger.info("PAINEL MACRO - ANALISE DO PAINEL MACROECONOMICO")
logger.info("=" * 80)

# --- contexto e EDA base ---
gerar_graficos_contexto_macro(df_desemprego, df_pib, df_ipca_ibge, df_inadt, painel_macro)

imprimir_visao_geral_macro(df_macro, COLS_BASE)
gerar_estatisticas_descritivas_macro(df_macro, COLS_BASE, COL_ALVO_MACRO)
analisar_dados_faltantes_macro(df_macro, COLS_BASE)
analisar_outliers_macro(df_macro, COLS_BASE)

gerar_boxplots_outliers_macro(df_macro, COLS_BASE, painel_macro)
gerar_distribuicoes_numericas_macro(df_macro, COLS_BASE, painel_macro)
gerar_distribuicoes_temporais(df_macro, COL_ALVO_MACRO, MESES_PT, painel_macro)
gerar_series_temporais_completas(df_macro, COLS_BASE, painel_macro)
gerar_distribuicao_inadimplencia(df_macro, COL_ALVO_MACRO, painel_macro)
gerar_bivariada_indicadores_macro(df_macro, COL_ALVO_MACRO, COLS_BASE, painel_macro)
gerar_inadimplencia_por_periodo(df_macro, COL_ALVO_MACRO, MESES_PT, painel_macro)
gerar_matriz_correlacao_macro(df_macro, COLS_BASE, COL_ALVO_MACRO, painel_macro)
gerar_analises_avancadas_macro(df_macro, df_pib, COL_ALVO_MACRO, MESES_PT, painel_macro)
insights_finais_macro(df_macro, COL_ALVO_MACRO, MESES_PT)

# --- novos graficos macro ---
gerar_pf_vs_pj(df_macro, painel_macro)
gerar_variacao_yoy(df_macro, COL_ALVO_MACRO, painel_macro)
gerar_carteira_vs_inadimplencia(df_macro, COL_ALVO_MACRO, painel_macro)
gerar_scatter_matrix_macro(df_macro, COLS_BASE, painel_macro)
gerar_analise_regimes(df_macro, COL_ALVO_MACRO, painel_macro)
gerar_decomposicao_stl(df_macro, COL_ALVO_MACRO, painel_macro)
gerar_acf_pacf(df_macro, COL_ALVO_MACRO, painel_macro)
gerar_coeficientes_ols(df_macro, COL_ALVO_MACRO, COLS_BASE, painel_macro)

# ===========================================================================
# PAINEL EDA  =>  outputs/painel_eda_inadimplencia.png
# ===========================================================================
if usar_olinda:
    logger.info("\n" + "=" * 80)
    logger.info("PAINEL EDA - ANALISE POR MODALIDADE/SEGMENTO (OLINDA)")
    logger.info("=" * 80)

    logger.info(f"\n[VISAO GERAL - {df_olinda.shape[0]:,} linhas x {df_olinda.shape[1]} colunas]")
    logger.info(df_olinda.dtypes.to_string())

    df_olinda = limpar_dados_inadimplencia_br(df_olinda, usando_macro=False)

    num_cols = [c for c in ['TaxaJurosAoMes', 'TaxaJurosAoAno', 'TaxaInadimplencia',
                             'NumeroDeContratos', 'BaseDeCalculo'] if c in df_olinda.columns]
    cat_cols = [c for c in ['Segmento', 'Modalidade', 'InstituicaoFinanceira']
                if c in df_olinda.columns]
    col_alvo_eda = 'TaxaInadimplencia' if 'TaxaInadimplencia' in df_olinda.columns else None

    analisar_outliers_br(df_olinda, num_cols)
    gerar_estatisticas_descritivas_br(df_olinda, num_cols, cat_cols)

    # --- EDA base Olinda ---
    gerar_boxplots_outliers(df_olinda, num_cols, painel_eda)
    gerar_distribuicoes_numericas(df_olinda, num_cols, painel_eda)
    gerar_distribuicoes_categoricas(df_olinda, cat_cols, False, painel_eda)
    gerar_graficos_variavel_alvo(df_olinda, col_alvo_eda, False, painel_eda)
    gerar_analises_bivariadas(df_olinda, col_alvo_eda, num_cols, cat_cols, False, painel_eda)
    gerar_analise_temporal(df_olinda, pd.DataFrame(), col_alvo_eda, False, painel_eda)
    gerar_matriz_correlacao(df_olinda, num_cols, pd.DataFrame(), False, col_alvo_eda, painel_eda)
    gerar_analise_avancada(df_olinda, col_alvo_eda, pd.DataFrame(), False, painel_eda)
    insights_finais_inadimplencia(df_olinda, col_alvo_eda, df_macro, False)

    # --- novos graficos Olinda ---
    gerar_ranking_instituicoes(df_olinda, col_alvo_eda, painel_eda)
    gerar_bubble_chart_modalidades(df_olinda, col_alvo_eda, painel_eda)
    gerar_evolucao_top_modalidades(df_olinda, col_alvo_eda, painel_eda)
    gerar_violin_por_segmento(df_olinda, col_alvo_eda, painel_eda)
    gerar_heatmap_instituicao_modalidade(df_olinda, col_alvo_eda, painel_eda)
    gerar_concentracao_mercado(df_olinda, painel_eda)
    gerar_quartis_modalidade_temporal(df_olinda, col_alvo_eda, painel_eda)
    gerar_juros_por_segmento(df_olinda, painel_eda)

# ---------------------------------------------------------------------------
# 4. FINALIZACAO
# ---------------------------------------------------------------------------
logger.info("\n" + "=" * 80)
logger.info("ANALISE CONCLUIDA COM SUCESSO!")
logger.info("=" * 80)

p = painel_macro.finalizar()
if p:
    logger.info(f"   Painel macro : {p.name}")

if usar_olinda:
    p = painel_eda.finalizar()
    if p:
        logger.info(f"   Painel EDA   : {p.name}")
else:
    logger.warning("   Painel EDA omitido (Olinda indisponivel).")

logger.info(f"   Pasta de saida: {output_dir}")
logger.info("=" * 80)

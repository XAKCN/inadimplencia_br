import warnings
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def configurar_exibicao(float_fmt='{:.2f}'):
    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    pd.set_option('display.float_format', float_fmt.format)
    pd.set_option('display.max_columns', None)


def montar_painel_sgs(series_sgs, series_ibge=None):
    """
    Faz merge das series SGS via outer join (cada serie tem seu proprio
    inicio historico) e das series IBGE via left join (preserva o
    backbone mensal do SGS). Series vazias sao ignoradas.
    """
    logger = logging.getLogger(__name__)

    validas = [s for s in series_sgs if not s.empty]
    if not validas:
        return pd.DataFrame(columns=['data'])

    df = validas[0].copy()
    for s in validas[1:]:
        df = pd.merge(df, s, on='data', how='outer')

    if series_ibge:
        for s in series_ibge:
            if not s.empty:
                df = pd.merge(df, s, on='data', how='left')

    df = df.sort_values('data').reset_index(drop=True)

    n_total = len(df)
    n_completo = df.notna().all(axis=1).sum()
    logger.info(f"   Painel SGS: {n_total} obs | {n_completo} com todas as colunas preenchidas")

    return df

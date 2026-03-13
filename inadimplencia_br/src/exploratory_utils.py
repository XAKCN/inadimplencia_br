import logging

import numpy as np
import pandas as pd
from scipy import stats

from .config import LIMIARES

logger = logging.getLogger(__name__)


def limpar_dados_inadimplencia_br(df, usando_macro=False):
    logger.info("\n" + "=" * 80)
    logger.info("3) LIMPEZA DE DADOS")
    logger.info("=" * 80)

    logger.info("\n[3.1 VALORES FALTANTES]")
    logger.info("-" * 50)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Quantidade': missing,
        'Porcentagem (%)': missing_pct.round(2)
    }).sort_values('Porcentagem (%)', ascending=False)
    missing_df = missing_df[missing_df['Quantidade'] > 0]

    if missing_df.empty:
        logger.info("   Nenhum valor faltante encontrado.")
    else:
        logger.info(missing_df.to_string())

    if not usando_macro:
        if 'TaxaInadimplencia' in df.columns:
            n_antes = len(df)
            df.dropna(subset=['TaxaInadimplencia'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info(f"\n   Removidas {n_antes - len(df):,} linhas sem TaxaInadimplencia.")

        # Taxas de juros: imputar com mediana (mais robusta que media para distribuicoes assimetricas)
        for col in ['TaxaJurosAoMes', 'TaxaJurosAoAno']:
            if col in df.columns and df[col].isnull().any():
                mediana = df[col].median()
                n_imp = df[col].isnull().sum()
                df[col].fillna(mediana, inplace=True)
                logger.info(f"   {col}: {n_imp} valores imputados com mediana ({mediana:.2f})")

        # Contagens: imputar com 0 (ausencia de registro = sem contratos naquele mes/modalidade)
        for col in ['NumeroDeContratos', 'BaseDeCalculo']:
            if col in df.columns and df[col].isnull().any():
                n_imp = df[col].isnull().sum()
                df[col].fillna(0, inplace=True)
                logger.info(f"   {col}: {n_imp} valores imputados com 0")

    logger.info(f"\n   Dataset apos limpeza: {len(df):,} linhas")
    return df


def analisar_outliers_br(df, num_cols):
    logger.info("\n[3.2 ANALISE DE OUTLIERS]")
    logger.info("-" * 50)

    fator = LIMIARES['outlier_iqr']
    for col in num_cols:
        serie = df[col].dropna()
        if len(serie) < 4:
            continue
        Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((serie < Q1 - fator * IQR) | (serie > Q3 + fator * IQR)).sum()
        logger.info(
            f"   {col:<28}: {n_out:>5} outliers ({n_out/len(serie)*100:.1f}%)"
            f"  |  IQR=[{Q1:.2f}, {Q3:.2f}]"
        )

    logger.info("""
   DECISAO: Outliers mantidos.
   Taxas de juros elevadas (ex.: cartao rotativo) e bases de calculo
   grandes sao parte da realidade do mercado brasileiro e carregam
   informacao essencial para analise de inadimplencia.
""")


def gerar_estatisticas_descritivas_br(df, num_cols, cat_cols=None):
    logger.info("\n" + "=" * 80)
    logger.info("4) ESTATISTICAS DESCRITIVAS")
    logger.info("=" * 80)

    logger.info("\n[4.1 VARIAVEIS NUMERICAS]")
    logger.info("-" * 50)
    if num_cols:
        desc = df[num_cols].describe().T
        desc['cv (%)'] = (desc['std'] / desc['mean'].abs() * 100).round(1)
        logger.info(desc.to_string())

    logger.info("\n[4.2 ESTATISTICAS ADICIONAIS]")
    logger.info("-" * 50)
    for col in num_cols:
        serie = df[col].dropna()
        if len(serie) < 2:
            continue
        sk = serie.skew()
        ku = serie.kurtosis()
        tipo_sk = "forte assimetria" if abs(sk) > 1 else "assimetria moderada" if abs(sk) > 0.5 else "aprox. simetrica"
        logger.info(f"\n   {col}:")
        logger.info(f"      Assimetria: {sk:+.3f} ({tipo_sk})")
        logger.info(f"      Curtose:    {ku:+.3f}")

    if cat_cols:
        logger.info("\n[4.3 VARIAVEIS CATEGORICAS]")
        logger.info("-" * 50)
        for col in cat_cols:
            n_cat = df[col].nunique()
            logger.info(f"\n   {col}: {n_cat} categorias unicas")
            top5 = df[col].value_counts().head(5)
            for cat, cnt in top5.items():
                logger.info(f"      {cat:<50}: {cnt:>6} ({cnt/len(df)*100:.1f}%)")


def imprimir_visao_geral_macro(df, cols_base):
    logger.info("\n" + "=" * 80)
    logger.info("2) VISAO GERAL DO PAINEL")
    logger.info("=" * 80)

    logger.info("\n[Primeiras 5 linhas:]")
    cols_to_print = ['data'] + cols_base
    logger.info(df[cols_to_print].head().to_string())

    logger.info("\n[Disponibilidade por coluna:]")
    for col in cols_base:
        n_ok = df[col].notna().sum()
        if n_ok > 0:
            primeiro = df.loc[df[col].notna(), 'data'].min()
            ultimo   = df.loc[df[col].notna(), 'data'].max()
            logger.info(f"   {col:<30}: {n_ok:>4} obs  |  {primeiro.date()} -> {ultimo.date()}")
        else:
            logger.info(f"   {col:<30}: {n_ok:>4} obs")


def gerar_estatisticas_descritivas_macro(df, cols_base, col_alvo):
    logger.info("\n" + "=" * 80)
    logger.info("3) ESTATISTICAS DESCRITIVAS")
    logger.info("=" * 80)

    desc = df[cols_base].describe().T
    desc['cv_pct']     = (desc['std'] / desc['mean'].abs() * 100).round(1)
    desc['assimetria'] = df[cols_base].skew().round(3)
    desc['curtose']    = df[cols_base].kurtosis().round(3)
    logger.info(desc.to_string())

    if col_alvo and col_alvo in df.columns:
        s_alvo = df[col_alvo].dropna()
        if not s_alvo.empty:
            logger.info(f"\n[DESTAQUE - {col_alvo}]")
            logger.info(
                f"   Minimo  : {s_alvo.min():.3f}%  em  "
                f"{df.loc[df[col_alvo] == s_alvo.min(), 'data'].iloc[0].date()}"
            )
            logger.info(
                f"   Maximo  : {s_alvo.max():.3f}%  em  "
                f"{df.loc[df[col_alvo] == s_alvo.max(), 'data'].iloc[0].date()}"
            )
            logger.info(f"   Amplitude: {s_alvo.max() - s_alvo.min():.3f} p.p.")


def analisar_dados_faltantes_macro(df, cols_base):
    logger.info("\n" + "=" * 80)
    logger.info("4) DADOS FALTANTES")
    logger.info("=" * 80)

    missing = df[cols_base].isnull().sum()
    missing_pct = missing / len(df) * 100
    miss_df = pd.DataFrame({'n': missing, '%': missing_pct.round(1)})
    miss_df = miss_df[miss_df['n'] > 0]
    if miss_df.empty:
        logger.info("   Nenhum valor faltante nas colunas base.")
    else:
        logger.info(miss_df.to_string())
        logger.info("\n   Nota: PIB per capita e anual - repetido para todos os meses do ano.")
        logger.info("   Nota: Desemprego PNAD e trimestral - interpolado linearmente.")


def analisar_outliers_macro(df, cols_base):
    logger.info("\n" + "=" * 80)
    logger.info("5) DETECCAO DE OUTLIERS")
    logger.info("=" * 80)

    fator = LIMIARES['outlier_iqr']
    z_max = LIMIARES['outlier_zscore']

    for col in cols_base:
        s = df[col].dropna()
        if len(s) < 4:
            continue
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((s < Q1 - fator * IQR) | (s > Q3 + fator * IQR)).sum()
        zscore_out = (np.abs(stats.zscore(s)) > z_max).sum()
        logger.info(
            f"   {col:<30}: IQR={n_out:>3} ({n_out/len(s)*100:.1f}%)  "
            f"Z>3={zscore_out:>3}  |  [{Q1:.3f}, {Q3:.3f}]"
        )

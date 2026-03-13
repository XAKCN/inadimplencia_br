import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .config import LIMIARES
from .constants import MESES_PT, CORES_TS

logger = logging.getLogger(__name__)


def gerar_graficos_contexto_macro(df_desemprego, df_pib, df_ipca_ibge, df_inadt, saida_graficos):
    logger.info("\n[1.4 GRAFICOS DE CONTEXTO MACROECONOMICO - IBGE]")
    logger.info("-" * 50)

    if not df_desemprego.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_desemprego['data'], df_desemprego['desemprego'],
                lw=2, color='#e67e22', label='Desocupacao')
        mm3 = df_desemprego['desemprego'].rolling(3, min_periods=1).mean()
        ax.plot(df_desemprego['data'], mm3,
                lw=2, ls='--', color='black', alpha=0.6, label='MM 3m')
        ax.fill_between(df_desemprego['data'], df_desemprego['desemprego'],
                        alpha=0.15, color='#e67e22')
        ax.set_title('Taxa de Desocupacao (PNAD Continua) - Brasil', fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel('Taxa (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        saida_graficos.registrar(fig, '01_desemprego_pnad.png',
                                 'Desemprego - Taxa de Desocupacao (PNAD Continua)')
        logger.info("   [OK] Desemprego IBGE gerado.")

    if not df_pib.empty:
        anos_pib = df_pib['data'].dt.year
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(anos_pib, df_pib['pib_per_capita'],
                      color='#27ae60', edgecolor='black', alpha=0.85)
        ax.set_title('PIB per Capita (R$ correntes) - Brasil', fontweight='bold')
        ax.set_xlabel('Ano')
        ax.set_ylabel('R$')
        ax.set_xticks(anos_pib)
        ax.set_xticklabels(anos_pib.astype(str))
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f'R${h/1000:.0f}k', ha='center', va='bottom', fontsize=8, rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        saida_graficos.registrar(fig, '02_pib_per_capita_anual.png',
                                 'PIB per Capita - Valores Anuais (R$)')
        logger.info("   [OK] PIB per capita IBGE gerado.")

    if not df_ipca_ibge.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_ipca_ibge['data'], df_ipca_ibge['ipca_mensal'],
                lw=1.5, color='#e74c3c', alpha=0.8, label='IPCA mensal')
        ax.fill_between(df_ipca_ibge['data'], df_ipca_ibge['ipca_mensal'], 0,
                        where=(df_ipca_ibge['ipca_mensal'] > 0),
                        color='#e74c3c', alpha=0.2)
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        mm12 = df_ipca_ibge['ipca_mensal'].rolling(12, min_periods=3).mean()
        ax.plot(df_ipca_ibge['data'], mm12,
                lw=2, color='#c0392b', ls='--', label='MM 12m')
        ax.set_title('IPCA - Variacao Mensal (%) - Brasil | IBGE Tabela 1737',
                     fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel('Variacao Mensal (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        saida_graficos.registrar(fig, '03_ipca_variacao_mensal.png',
                                 'IPCA - Variacao Mensal (%)')
        logger.info("   [OK] IPCA mensal IBGE gerado.")

    # Painel comparativo: Desemprego + IPCA + Inadimplencia
    _colunas_painel = {
        'desemprego':          ('df_desemprego', '#e67e22', 'Desocupacao (%)'),
        'ipca_mensal':         ('df_ipca_ibge',  '#e74c3c', 'IPCA Mensal (%)'),
        'inadimplencia_total': ('df_inadt',       '#2980b9', 'Inadimplencia Total (%)'),
    }
    _dfs_painel = {
        'df_desemprego': df_desemprego,
        'df_ipca_ibge':  df_ipca_ibge,
        'df_inadt':      df_inadt,
    }
    _series_disponiveis = [col for col, meta in _colunas_painel.items()
                           if not _dfs_painel[meta[0]].empty]

    if len(_series_disponiveis) >= 2:
        fig, axes = plt.subplots(len(_series_disponiveis), 1,
                                 figsize=(16, 3.5 * len(_series_disponiveis)),
                                 sharex=True)
        fig.suptitle('Contexto Macroeconomico: IBGE + BACEN', fontsize=13, fontweight='bold')
        axes = np.atleast_1d(axes).ravel()

        for idx, col in enumerate(_series_disponiveis):
            src_key  = _colunas_painel[col][0]
            cor      = _colunas_painel[col][1]
            lbl      = _colunas_painel[col][2]
            df_src   = _dfs_painel[src_key]
            col_name = col if col in df_src.columns else df_src.columns[1]

            axes[idx].plot(df_src['data'], df_src[col_name], lw=2, color=cor, label=lbl)
            axes[idx].fill_between(df_src['data'], df_src[col_name], alpha=0.12, color=cor)
            axes[idx].set_ylabel(lbl, fontsize=9)
            axes[idx].legend(fontsize=8, loc='upper left')
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Data')
        plt.tight_layout()
        saida_graficos.registrar(fig, '04_contexto_macroeconomico.png',
                                 'Contexto Macroeconomico - Desemprego, IPCA e Inadimplencia')
        logger.info("   [OK] Painel comparativo macro gerado.")


def gerar_boxplots_outliers_macro(df, cols_base, saida_graficos):
    if cols_base:
        n_c = len(cols_base)
        nr  = (n_c + 2) // 3
        fig, axes = plt.subplots(nr, 3, figsize=(18, nr * 4))
        fig.suptitle('Boxplots das Variaveis Macroeconomicas', fontsize=14, fontweight='bold')
        axes = np.atleast_1d(axes).ravel()

        fator = LIMIARES['outlier_iqr']
        for idx, col in enumerate(cols_base):
            s = df[col].dropna()
            axes[idx].boxplot(s, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='#4a90d9', alpha=0.65))
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            axes[idx].axhline(Q3 + fator * IQR, color='red', ls='--', lw=1, alpha=0.7)
            axes[idx].axhline(Q1 - fator * IQR, color='red', ls='--', lw=1, alpha=0.7)
            axes[idx].set_title(col, fontsize=9)
            axes[idx].set_ylabel('Valor')

        for ax in axes[n_c:]:
            ax.set_visible(False)

        plt.tight_layout()
        saida_graficos.registrar(fig, '05_outliers_boxplots.png',
                                 'Outliers - Analise por Boxplot')


def gerar_distribuicoes_numericas_macro(df, cols_base, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("6) DISTRIBUICAO DAS VARIAVEIS NUMERICAS")
    logger.info("=" * 80)

    n_c = len(cols_base)
    nr  = (n_c + 2) // 3
    fig, axes = plt.subplots(nr, 3, figsize=(18, nr * 4))
    fig.suptitle('Distribuicao das Variaveis Macroeconomicas', fontsize=14, fontweight='bold')
    axes = np.atleast_1d(axes).ravel()

    for idx, col in enumerate(cols_base):
        s = df[col].dropna()
        sns.histplot(s, kde=True, ax=axes[idx], bins=25, color='steelblue')
        axes[idx].axvline(s.mean(),   color='red',   ls='--', lw=1.5,
                          label=f'Media {s.mean():.2f}')
        axes[idx].axvline(s.median(), color='green', ls=':',  lw=1.5,
                          label=f'Mediana {s.median():.2f}')
        axes[idx].set_title(col, fontsize=9)
        axes[idx].legend(fontsize=7)

    for ax in axes[n_c:]:
        ax.set_visible(False)

    plt.tight_layout()
    saida_graficos.registrar(fig, '06_distribuicao_variaveis_numericas.png',
                             'Distribuicao das Variaveis Numericas')
    logger.info("   [OK] Histogramas gerados.")


def gerar_distribuicoes_temporais(df, col_alvo, meses_pt, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("7) DISTRIBUICAO POR VARIAVEIS TEMPORAIS")
    logger.info("=" * 80)

    if col_alvo:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f'Distribuicao Temporal - {col_alvo}', fontsize=13, fontweight='bold')

        media_ano = df.groupby('ano')[col_alvo].mean()
        bars = axes[0].bar(media_ano.index, media_ano.values,
                           color='coral', edgecolor='black')
        axes[0].set_title(f'Media Anual - {col_alvo}')
        axes[0].set_xlabel('Ano')
        axes[0].set_ylabel('Taxa Media (%)')
        axes[0].set_xticks(media_ano.index)
        axes[0].set_xticklabels(media_ano.index.astype(str), rotation=45)
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, h,
                         f'{h:.2f}%', ha='center', va='bottom', fontsize=8)

        media_mes = df.groupby('mes')[col_alvo].mean().reindex(range(1, 13))
        axes[1].bar(range(1, 13), media_mes.values, color='lightgreen', edgecolor='black')
        axes[1].set_title('Media por Mes (Sazonalidade)')
        axes[1].set_xlabel('Mes')
        axes[1].set_ylabel('Taxa Media (%)')
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(meses_pt, rotation=45)

        plt.tight_layout()
        saida_graficos.registrar(fig, '07_distribuicao_temporal_media.png',
                                 'Distribuicao Temporal - Media Anual e Sazonal')
        logger.info("   [OK] Graficos temporais gerados.")


def gerar_series_temporais_completas(df, cols_base, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("8) SERIES TEMPORAIS - TODOS OS INDICADORES")
    logger.info("=" * 80)

    cols_ts = [c for c in cols_base if c != 'pib_per_capita']  # PIB e anual, nao plota serie mensal
    if cols_ts:
        fig, axes = plt.subplots(len(cols_ts), 1, figsize=(16, 3.5 * len(cols_ts)))
        fig.suptitle('Series Temporais - Indicadores Macroeconomicos',
                     fontsize=14, fontweight='bold')
        axes = np.atleast_1d(axes).ravel()

        for idx, col in enumerate(cols_ts):
            df_col = df[['data', col]].dropna()
            axes[idx].plot(df_col['data'], df_col[col], lw=1.8,
                           color=CORES_TS[idx % len(CORES_TS)], label=col)
            mm12 = df_col[col].rolling(12, min_periods=3).mean()
            axes[idx].plot(df_col['data'], mm12, lw=2.0, ls='--',
                           color='black', alpha=0.5, label='MM 12m')
            axes[idx].set_ylabel(col, fontsize=9)
            axes[idx].legend(fontsize=8, loc='upper left')
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Data')
        plt.tight_layout()
        saida_graficos.registrar(fig, '08_series_temporais_completas.png',
                                 'Series Temporais Completas (BACEN + IBGE)')
        logger.info("   [OK] Series temporais geradas.")


def gerar_distribuicao_inadimplencia(df, col_alvo, saida_graficos):
    if col_alvo:
        logger.info("\n" + "=" * 80)
        logger.info("9) CLASSIFICACAO DE PERIODOS DE INADIMPLENCIA")
        logger.info("=" * 80)

        s_alvo = df[col_alvo].dropna()
        q25, mediana, q75 = s_alvo.quantile([0.25, 0.5, 0.75])

        logger.info(f"\n   Quartis:  Q1={q25:.3f}%  |  Mediana={mediana:.3f}%  |  Q3={q75:.3f}%")

        bins         = [-np.inf, q25, mediana, q75, np.inf]
        labels_class = ['Baixa', 'Media-Baixa', 'Media-Alta', 'Alta']
        df['classificacao'] = pd.cut(df[col_alvo], bins=bins, labels=labels_class)

        contagens = df['classificacao'].value_counts().reindex(labels_class)
        for nivel, cnt in contagens.items():
            if pd.notna(cnt):
                logger.info(f"   {nivel:<15}: {int(cnt):>4} meses ({int(cnt)/len(df)*100:.1f}%)")

        cores_nivel = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        cnt_vals    = [contagens.get(lbl, 0) for lbl in labels_class]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Distribuicao da Variavel Alvo: {col_alvo}',
                     fontsize=13, fontweight='bold')

        sns.histplot(s_alvo, kde=True, ax=axes[0], bins=30, color='#4a90d9')
        axes[0].axvline(s_alvo.mean(),   color='red',   ls='--', lw=1.5,
                        label=f'Media {s_alvo.mean():.2f}%')
        axes[0].axvline(s_alvo.median(), color='green', ls=':',  lw=1.5,
                        label=f'Mediana {s_alvo.median():.2f}%')
        axes[0].set_title('Histograma com KDE')
        axes[0].set_xlabel('Taxa (%)')
        axes[0].legend(fontsize=8)

        bars = axes[1].bar(labels_class, cnt_vals, color=cores_nivel, edgecolor='black')
        axes[1].set_title('Periodos por Nivel')
        axes[1].set_ylabel('Qtde de meses')
        for bar, v in zip(bars, cnt_vals):
            if v > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                             f'{int(v)}\n({int(v)/len(df)*100:.1f}%)',
                             ha='center', va='bottom', fontsize=8)

        cnt_valid    = [v for v in cnt_vals if v > 0]
        labels_valid = [lbl for lbl, v in zip(labels_class, cnt_vals) if v > 0]
        cores_valid  = [c for c, v in zip(cores_nivel, cnt_vals) if v > 0]
        axes[2].pie(cnt_valid, labels=labels_valid, autopct='%1.1f%%',
                    colors=cores_valid, startangle=90, explode=[0.03] * len(cnt_valid))
        axes[2].set_title('Proporcao por Nivel')

        plt.tight_layout()
        saida_graficos.registrar(fig, '09_distribuicao_inadimplencia.png',
                                 'Distribuicao da Taxa de Inadimplencia')
        logger.info("   [OK] Graficos de classificacao gerados.")


def gerar_bivariada_indicadores_macro(df, col_alvo, cols_base, saida_graficos):
    if col_alvo:
        logger.info("\n" + "=" * 80)
        logger.info("10) ANALISE BIVARIADA - INADIMPLENCIA vs INDICADORES MACRO")
        logger.info("=" * 80)

        outros = [c for c in cols_base if c != col_alvo and c != 'pib_per_capita']
        n_out  = len(outros)

        if n_out > 0:
            nr = (n_out + 2) // 3
            fig, axes = plt.subplots(nr, 3, figsize=(18, nr * 5))
            fig.suptitle(f'Scatter: {col_alvo} vs Indicadores Macro',
                         fontsize=13, fontweight='bold')
            axes = np.atleast_1d(axes).ravel()

            for idx, col in enumerate(outros):
                df_s = df[[col_alvo, col, 'ano']].dropna()
                if len(df_s) < 5:
                    logger.warning(
                        f"   [AVISO] {col}: apenas {len(df_s)} amostras - scatter omitido."
                    )
                    axes[idx].set_visible(False)
                    continue
                sc = axes[idx].scatter(df_s[col], df_s[col_alvo],
                                       c=df_s['ano'], cmap='viridis', alpha=0.6, s=25)
                m, b, r, p, _ = stats.linregress(df_s[col], df_s[col_alvo])
                xr = np.linspace(df_s[col].min(), df_s[col].max(), 100)
                axes[idx].plot(xr, m * xr + b, 'r-', lw=2, label=f'r={r:.2f} (p={p:.3f})')
                axes[idx].set_xlabel(col, fontsize=9)
                axes[idx].set_ylabel(col_alvo, fontsize=9)
                axes[idx].set_title(f'{col_alvo} vs {col}', fontsize=9)
                axes[idx].legend(fontsize=7)
                plt.colorbar(sc, ax=axes[idx], label='Ano', shrink=0.8)
                logger.info(f"   {col_alvo} vs {col:<28}: r={r:+.3f}  p={p:.4f}")

            for ax in axes[n_out:]:
                ax.set_visible(False)

            plt.tight_layout()
            saida_graficos.registrar(fig, '10_inadimplencia_vs_indicadores_macro.png',
                                     'Inadimplencia vs Indicadores Macroeconomicos')
            logger.info("   [OK] Scatter plots bivariados gerados.")


def gerar_inadimplencia_por_periodo(df, col_alvo, meses_pt, saida_graficos):
    if col_alvo:
        logger.info("\n" + "=" * 80)
        logger.info("11) INADIMPLENCIA POR VARIAVEIS TEMPORAIS")
        logger.info("=" * 80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{col_alvo} por Variaveis Temporais', fontsize=13, fontweight='bold')

        df_a = df[['data', 'ano', 'mes', 'trimestre', col_alvo]].dropna()

        df_a.boxplot(column=col_alvo, by='ano', ax=axes[0, 0])
        axes[0, 0].set_title('Distribuicao por Ano')
        axes[0, 0].set_xlabel('Ano')
        axes[0, 0].set_ylabel('Taxa (%)')
        plt.sca(axes[0, 0])
        plt.xticks(rotation=45)
        # Restaura suptitle sobrescrito pelo boxplot
        fig.suptitle(f'{col_alvo} por Variaveis Temporais', fontsize=13, fontweight='bold')

        media_mes = df_a.groupby('mes')[col_alvo].mean().reindex(range(1, 13))
        axes[0, 1].bar(range(1, 13), media_mes.values, color='steelblue', edgecolor='black')
        axes[0, 1].set_title('Sazonalidade - Media por Mes')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(meses_pt, rotation=45)
        axes[0, 1].set_ylabel('Taxa Media (%)')

        axes[0, 2].set_visible(False)

        cores_ano = plt.cm.tab10(np.linspace(0, 1, df_a['ano'].nunique()))
        for i, ano in enumerate(sorted(df_a['ano'].unique())):
            df_ano = df_a[df_a['ano'] == ano]
            axes[1, 0].plot(df_ano['data'], df_ano[col_alvo],
                            color=cores_ano[i], lw=1.8, label=str(ano))
        axes[1, 0].set_title('Serie por Ano')
        axes[1, 0].set_xlabel('Data')
        axes[1, 0].set_ylabel('Taxa (%)')
        axes[1, 0].legend(ncol=3, fontsize=7, loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)

        if 'inadt_var_pp' in df.columns:
            df_v = df[['data', 'inadt_var_pp']].dropna()
            axes[1, 1].plot(df_v['data'], df_v['inadt_var_pp'], lw=1, color='gray', alpha=0.8)
            axes[1, 1].fill_between(df_v['data'], df_v['inadt_var_pp'], 0,
                                    where=(df_v['inadt_var_pp'] > 0), color='#e74c3c', alpha=0.4)
            axes[1, 1].fill_between(df_v['data'], df_v['inadt_var_pp'], 0,
                                    where=(df_v['inadt_var_pp'] < 0), color='#27ae60', alpha=0.4)
            axes[1, 1].axhline(0, color='black', lw=0.8, ls='--')
            axes[1, 1].set_title('Variacao Mensal (p.p.)')
            axes[1, 1].set_xlabel('Data')
            axes[1, 1].set_ylabel('Variacao (p.p.)')
            axes[1, 1].grid(True, alpha=0.3)

        if 'inadt_mm12' in df.columns:
            df_mm = df[['data', col_alvo, 'inadt_mm12']].dropna()
            axes[1, 2].plot(df_mm['data'], df_mm[col_alvo], lw=1.5, alpha=0.6, label='Mensal')
            axes[1, 2].plot(df_mm['data'], df_mm['inadt_mm12'], lw=2.5,
                            color='#c0392b', label='MM 12m')
            axes[1, 2].set_title('Mensal vs Media Movel 12m')
            axes[1, 2].set_xlabel('Data')
            axes[1, 2].set_ylabel('Taxa (%)')
            axes[1, 2].legend(fontsize=8)
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        saida_graficos.registrar(fig, '11_inadimplencia_por_periodo.png',
                                 'Inadimplencia por Periodo - Analise Temporal')
        logger.info("   [OK] Graficos temporais por categoria gerados.")


def gerar_matriz_correlacao_macro(df, cols_base, col_alvo, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("12) MATRIZ DE CORRELACAO")
    logger.info("=" * 80)

    cols_corr = cols_base.copy()
    for lag in ['inadt_lag1', 'inadt_lag3', 'inadt_lag6', 'inadt_lag12',
                'inadt_mm3', 'inadt_mm6', 'inadt_mm12']:
        if lag in df.columns:
            cols_corr.append(lag)

    df_corr = df[cols_corr].dropna()
    if df_corr.shape[1] >= 2 and len(df_corr) >= 5:
        corr_matrix = df_corr.corr()
        sz = max(10, len(corr_matrix))
        fig = plt.figure(figsize=(sz, sz - 1))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, square=True, linewidths=0.5,
                    vmin=-1, vmax=1, cbar_kws={'shrink': 0.7})
        plt.title('Matriz de Correlacao - Painel Macro', fontsize=13, fontweight='bold')
        plt.tight_layout()
        saida_graficos.registrar(fig, '12_matriz_correlacao.png', 'Matriz de Correlacao')
        logger.info("   [OK] Matriz gerada.")

        if col_alvo and col_alvo in corr_matrix.columns:
            corr_alvo = (corr_matrix[col_alvo]
                         .drop(col_alvo, errors='ignore')
                         .sort_values(key=abs, ascending=False))
            logger.info(f"\n   Correlacoes com {col_alvo}:")
            for var, r in corr_alvo.items():
                forca = ("forte" if abs(r) > LIMIARES['corr_forte']
                         else "moderada" if abs(r) > LIMIARES['corr_moderada']
                         else "fraca")
                logger.info(f"      {var:<30}: {r:+.3f}  ({forca})")


def gerar_analises_avancadas_macro(df, df_pib, col_alvo, meses_pt, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("13) ANALISE DE RELACOES E PADROES AVANCADOS")
    logger.info("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analise de Relacoes Avancadas', fontsize=13, fontweight='bold')

    if col_alvo and 'inadt_lag1' in df.columns:
        df_ac = df[[col_alvo, 'inadt_lag1', 'ano']].dropna()
        sc = axes[0, 0].scatter(df_ac['inadt_lag1'], df_ac[col_alvo],
                                c=df_ac['ano'], cmap='viridis', alpha=0.7, s=40)
        axes[0, 0].plot([df_ac['inadt_lag1'].min(), df_ac['inadt_lag1'].max()],
                        [df_ac['inadt_lag1'].min(), df_ac['inadt_lag1'].max()],
                        'r--', alpha=0.4, label='y=x')
        axes[0, 0].set_xlabel('Taxa Mes Anterior (%)')
        axes[0, 0].set_ylabel('Taxa Atual (%)')
        axes[0, 0].set_title('Autocorrelacao (Lag 1)')
        axes[0, 0].legend(fontsize=8)
        plt.colorbar(sc, ax=axes[0, 0], label='Ano', shrink=0.8)

    if col_alvo and 'inadt_mm3' in df.columns and 'inadt_mm12' in df.columns:
        df_mm = df[['inadt_mm3', 'inadt_mm12', 'ano']].dropna()
        sc = axes[0, 1].scatter(df_mm['inadt_mm3'], df_mm['inadt_mm12'],
                                c=df_mm['ano'], cmap='tab10', alpha=0.7, s=35)
        axes[0, 1].set_xlabel('Media Movel 3m (%)')
        axes[0, 1].set_ylabel('Media Movel 12m (%)')
        axes[0, 1].set_title('MM 3m vs MM 12m (por ano)')
        plt.colorbar(sc, ax=axes[0, 1], label='Ano', shrink=0.8)

    if col_alvo:
        df_hm = df[['ano', 'mes', col_alvo]].dropna()
        pivot = df_hm.pivot_table(values=col_alvo, index='ano', columns='mes', aggfunc='mean')
        pivot.columns = meses_pt[:len(pivot.columns)]
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    ax=axes[1, 0], linewidths=0.4, cbar_kws={'shrink': 0.8})
        axes[1, 0].set_title(f'Heatmap: {col_alvo} (Ano x Mes)')
        axes[1, 0].set_xlabel('Mes')
        axes[1, 0].set_ylabel('Ano')

    if col_alvo:
        df_vi = df[['ano', col_alvo]].dropna()
        anos_uniq   = sorted(df_vi['ano'].unique())
        dados_violin = [df_vi[df_vi['ano'] == a][col_alvo].values for a in anos_uniq]
        dados_violin = [d for d in dados_violin if len(d) > 1]
        if dados_violin:
            axes[1, 1].violinplot(dados_violin, positions=range(len(dados_violin)),
                                  showmeans=True, showmedians=True)
            axes[1, 1].set_xticks(range(len(anos_uniq)))
            axes[1, 1].set_xticklabels([str(a) for a in anos_uniq], rotation=45)
            axes[1, 1].set_title(f'{col_alvo} - Violin por Ano')
            axes[1, 1].set_ylabel('Taxa (%)')
            axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    saida_graficos.registrar(fig, '13_relacoes_padroes_avancados.png',
                             'Relacoes e Padroes Avancados')
    logger.info("   [OK] Analise avancada gerada.")

    if not df_pib.empty:
        anos_pib = df_pib['data'].dt.year
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        bars = ax.bar(anos_pib, df_pib['pib_per_capita'],
                      color='#3498db', edgecolor='black')
        ax.set_title('PIB per Capita (R$ correntes) - IBGE', fontweight='bold')
        ax.set_xlabel('Ano')
        ax.set_ylabel('PIB per Capita (R$)')
        ax.set_xticks(anos_pib)
        ax.set_xticklabels(anos_pib.astype(str), rotation=45)
        for bar, (_, row) in zip(bars, df_pib.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, row['pib_per_capita'],
                    f'R$ {row["pib_per_capita"]:,.0f}',
                    ha='center', va='bottom', fontsize=7, rotation=45)
        plt.tight_layout()
        saida_graficos.registrar(fig, '14_pib_per_capita_evolucao.png',
                                 'PIB per Capita - Evolucao Historica')
        logger.info("   [OK] Grafico PIB per capita gerado.")


def insights_finais_macro(df, col_alvo, meses_pt):
    logger.info("\n" + "=" * 80)
    logger.info("14) INSIGHTS FINAIS")
    logger.info("=" * 80)

    if col_alvo:
        s          = df[col_alvo].dropna()
        media_ano  = df.groupby('ano')[col_alvo].mean()
        ano_max    = media_ano.idxmax()
        ano_min    = media_ano.idxmin()
        mes_max    = df.groupby('mes')[col_alvo].mean().idxmax()
        mes_min    = df.groupby('mes')[col_alvo].mean().idxmin()

        logger.info(f"""
   VARIAVEL ALVO  : {col_alvo}
   Periodo        : {df.loc[df[col_alvo].notna(), 'data'].min().date()} -> {df.loc[df[col_alvo].notna(), 'data'].max().date()}
   Media geral    : {s.mean():.3f}%
   Mediana        : {s.median():.3f}%
   Amplitude      : {s.max() - s.min():.3f} p.p.
   Maior media anual : {ano_max}  ({media_ano[ano_max]:.3f}%)
   Menor media anual : {ano_min}  ({media_ano[ano_min]:.3f}%)
   Sazonalidade   : maior em {meses_pt[mes_max-1]} | menor em {meses_pt[mes_min-1]}
""")

    logger.info("""
   ACHADOS RELEVANTES:
   \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
   1. FONTES: 100% dados publicos verificaveis (BACEN SGS + IBGE SIDRA).

   2. MACRO-CORRELACOES: Selic e IPCA elevados tendem a estar associados
      a maiores taxas de inadimplencia com defasagem de 3-6 meses.
      Desemprego alto correlaciona positivamente com inadimplencia PF.

   3. INERCIA TEMPORAL: Forte autocorrelacao (lag 1 ~ 0.95+). A serie
      e altamente persistente - o melhor preditor do proximo mes e o
      valor atual. Medias moveis sao excelentes indicadores de tendencia.

   4. SAZONALIDADE: Observar o heatmap Ano x Mes para identificar
      padroes mensais recorrentes (tipicamente Jan e Fev mais altos
      em funcao de festas de fim de ano e carnaval;
      Dez mais baixo pelo 13o salario).

   5. PIB: Crescimento do PIB per capita correlaciona negativamente
      com inadimplencia no longo prazo - renda real crescente reduz
      o estresse financeiro das familias.

   6. PROXIMO PASSO: usar este painel como features exogenas em modelos
      de series temporais (ARIMAX, SARIMAX, Prophet ou XGBoost temporal).
""")


def gerar_pf_vs_pj(df, saida_graficos):
    cols = [c for c in ['inadimplencia_pf', 'inadimplencia_pj'] if c in df.columns]
    if len(cols) < 2 or 'data' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) PF vs PJ - SPREAD E RATIO")
    logger.info("=" * 80)

    df_pv = df[['data'] + cols].dropna()
    df_pv = df_pv.copy()
    df_pv['spread'] = df_pv['inadimplencia_pf'] - df_pv['inadimplencia_pj']
    df_pv['ratio']  = df_pv['inadimplencia_pf'] / df_pv['inadimplencia_pj'].replace(0, np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('Inadimplencia PF vs PJ — Comparativo', fontsize=13, fontweight='bold')

    axes[0].plot(df_pv['data'], df_pv['inadimplencia_pf'],
                 lw=2, color='#e74c3c', label='PF')
    axes[0].plot(df_pv['data'], df_pv['inadimplencia_pj'],
                 lw=2, color='#2980b9', label='PJ')
    axes[0].set_ylabel('Taxa (%)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(df_pv['data'], df_pv['spread'], 0,
                         where=(df_pv['spread'] > 0), color='#e74c3c', alpha=0.4, label='PF > PJ')
    axes[1].fill_between(df_pv['data'], df_pv['spread'], 0,
                         where=(df_pv['spread'] < 0), color='#2980b9', alpha=0.4, label='PJ > PF')
    axes[1].plot(df_pv['data'], df_pv['spread'], lw=1, color='gray', alpha=0.7)
    axes[1].axhline(0, color='black', lw=0.8, ls='--')
    axes[1].set_ylabel('Spread PF - PJ (p.p.)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_pv['data'], df_pv['ratio'], lw=2, color='#8e44ad')
    axes[2].axhline(1, color='black', lw=0.8, ls='--', alpha=0.5)
    axes[2].set_ylabel('Ratio PF / PJ')
    axes[2].set_xlabel('Data')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_01_pf_vs_pj.png',
                             'Inadimplencia PF vs PJ - Spread e Ratio')
    logger.info("   [OK] PF vs PJ gerado.")


def gerar_variacao_yoy(df, col_alvo, saida_graficos):
    if not col_alvo or col_alvo not in df.columns or 'data' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) VARIACAO ANO A ANO (YoY)")
    logger.info("=" * 80)

    df_y = df[['data', col_alvo]].dropna().copy()
    df_y = df_y.sort_values('data').reset_index(drop=True)
    df_y['yoy_pp']  = df_y[col_alvo] - df_y[col_alvo].shift(12)
    df_y['yoy_pct'] = df_y[col_alvo].pct_change(12) * 100
    df_y = df_y.dropna(subset=['yoy_pp'])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(f'Variacao Ano a Ano (YoY) — {col_alvo}',
                 fontsize=13, fontweight='bold')

    axes[0].plot(df_y['data'], df_y[col_alvo], lw=2, color='#2980b9', label='Nivel')
    axes[0].set_ylabel('Taxa (%)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(df_y['data'], df_y['yoy_pp'],
                color=np.where(df_y['yoy_pp'] > 0, '#e74c3c', '#27ae60'),
                width=20, alpha=0.8)
    axes[1].axhline(0, color='black', lw=0.8, ls='--')
    axes[1].set_ylabel('Variacao YoY (p.p.)')
    axes[1].set_xlabel('Data')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_02_variacao_yoy.png',
                             'Variacao Ano a Ano (YoY)')
    logger.info("   [OK] Variacao YoY gerada.")


def gerar_carteira_vs_inadimplencia(df, col_alvo, saida_graficos):
    if (not col_alvo or col_alvo not in df.columns
            or 'saldo_credito_bi' not in df.columns
            or 'data' not in df.columns):
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) CARTEIRA DE CREDITO vs INADIMPLENCIA")
    logger.info("=" * 80)

    df_c = df[['data', col_alvo, 'saldo_credito_bi']].dropna()

    fig, ax1 = plt.subplots(figsize=(16, 5))
    fig.suptitle('Saldo da Carteira de Credito vs Taxa de Inadimplencia',
                 fontsize=13, fontweight='bold')

    ax2 = ax1.twinx()
    ax1.bar(df_c['data'], df_c['saldo_credito_bi'],
            width=20, color='#3498db', alpha=0.4, label='Saldo credito (R$ bi)')
    ax2.plot(df_c['data'], df_c[col_alvo],
             lw=2.5, color='#e74c3c', label='Inadimplencia (%)')

    ax1.set_xlabel('Data')
    ax1.set_ylabel('Saldo de Credito (R$ bi)', color='#3498db')
    ax2.set_ylabel('Taxa de Inadimplencia (%)', color='#e74c3c')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_03_carteira_vs_inadimplencia.png',
                             'Carteira de Credito vs Inadimplencia')
    logger.info("   [OK] Carteira vs inadimplencia gerado.")


def gerar_scatter_matrix_macro(df, cols_base, saida_graficos):
    cols_sm = [c for c in cols_base if c in df.columns][:6]
    if len(cols_sm) < 3:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) SCATTER MATRIX (PAIRPLOT) MACRO")
    logger.info("=" * 80)

    df_sm = df[cols_sm + (['ano'] if 'ano' in df.columns else [])].dropna()

    if 'ano' in df_sm.columns:
        norm = plt.Normalize(df_sm['ano'].min(), df_sm['ano'].max())
        cmap = plt.cm.viridis
        hue_col = 'ano'
        df_plot = df_sm.drop(columns='ano')
    else:
        hue_col = None
        df_plot = df_sm

    g = sns.pairplot(df_sm[cols_sm + ([hue_col] if hue_col else [])],
                     vars=cols_sm,
                     plot_kws={'alpha': 0.35, 's': 18},
                     diag_kind='kde',
                     corner=True)
    g.figure.suptitle('Scatter Matrix — Variaveis Macroeconomicas',
                       fontsize=12, fontweight='bold', y=1.01)

    saida_graficos.registrar(g.figure, 'extra_04_scatter_matrix.png',
                             'Scatter Matrix - Variaveis Macro')
    logger.info("   [OK] Scatter matrix gerada.")


def gerar_analise_regimes(df, col_alvo, saida_graficos):
    if not col_alvo or col_alvo not in df.columns or 'data' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) ANALISE DE REGIMES DE INADIMPLENCIA")
    logger.info("=" * 80)

    df_r = df[['data', col_alvo]].dropna().copy()
    q33 = df_r[col_alvo].quantile(0.33)
    q66 = df_r[col_alvo].quantile(0.66)

    def classifica(v):
        if v <= q33:
            return 'Baixa'
        elif v <= q66:
            return 'Media'
        return 'Alta'

    df_r['regime'] = df_r[col_alvo].apply(classifica)
    cores_regime = {'Baixa': '#27ae60', 'Media': '#f1c40f', 'Alta': '#e74c3c'}

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(f'Regimes de Inadimplencia — {col_alvo}',
                 fontsize=13, fontweight='bold')

    for regime, cor in cores_regime.items():
        mask = df_r['regime'] == regime
        ax.fill_between(df_r['data'], df_r[col_alvo],
                        where=mask, color=cor, alpha=0.35, label=regime)

    ax.plot(df_r['data'], df_r[col_alvo], lw=2, color='black', alpha=0.7)
    ax.axhline(q33, color='#27ae60', lw=1, ls='--', alpha=0.6)
    ax.axhline(q66, color='#e74c3c', lw=1, ls='--', alpha=0.6)
    ax.set_ylabel('Taxa (%)')
    ax.set_xlabel('Data')
    ax.legend(title='Regime', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_05_regimes_inadimplencia.png',
                             'Regimes de Inadimplencia (Baixa / Media / Alta)')
    logger.info("   [OK] Analise de regimes gerada.")


def gerar_decomposicao_stl(df, col_alvo, saida_graficos):
    if not col_alvo or col_alvo not in df.columns or 'data' not in df.columns:
        return

    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        logger.warning("   [AVISO] statsmodels nao instalado — STL omitido.")
        return

    s = df.set_index('data')[col_alvo].dropna().sort_index()
    if len(s) < 24:
        logger.warning("   [AVISO] Serie muito curta para STL (< 24 obs).")
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) DECOMPOSICAO STL")
    logger.info("=" * 80)

    stl = STL(s, period=12, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'Decomposicao STL — {col_alvo}', fontsize=13, fontweight='bold')

    for ax, serie, titulo, cor in zip(
        axes,
        [s, res.trend, res.seasonal, res.resid],
        ['Observado', 'Tendencia', 'Sazonalidade', 'Residuo'],
        ['#2980b9', '#e74c3c', '#27ae60', '#95a5a6'],
    ):
        ax.plot(serie.index, serie.values, lw=1.8, color=cor)
        ax.set_ylabel(titulo, fontsize=9)
        ax.grid(True, alpha=0.3)
        if titulo == 'Residuo':
            ax.axhline(0, color='black', lw=0.8, ls='--')

    axes[-1].set_xlabel('Data')
    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_06_decomposicao_stl.png',
                             'Decomposicao STL - Tendencia e Sazonalidade')
    logger.info("   [OK] Decomposicao STL gerada.")


def gerar_acf_pacf(df, col_alvo, saida_graficos):
    if not col_alvo or col_alvo not in df.columns:
        return

    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        logger.warning("   [AVISO] statsmodels nao instalado — ACF/PACF omitido.")
        return

    s = df[col_alvo].dropna()
    if len(s) < 30:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) ACF / PACF")
    logger.info("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'Autocorrelacao e Autocorrelacao Parcial — {col_alvo}',
                 fontsize=13, fontweight='bold')

    plot_acf(s, lags=24, ax=axes[0], color='#2980b9', title='ACF (ate lag 24)')
    plot_pacf(s, lags=24, ax=axes[1], color='#e74c3c', method='ywm',
              title='PACF (ate lag 24)')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_07_acf_pacf.png',
                             'Autocorrelacao (ACF) e Autocorrelacao Parcial (PACF)')
    logger.info("   [OK] ACF/PACF gerado.")


def gerar_coeficientes_ols(df, col_alvo, cols_base, saida_graficos):
    if not col_alvo or col_alvo not in df.columns:
        return

    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("   [AVISO] statsmodels nao instalado — OLS omitido.")
        return

    regressores = [c for c in cols_base
                   if c != col_alvo and c not in ('pib_per_capita',)
                   and c in df.columns]
    if not regressores:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) REGRESSAO OLS — COEFICIENTES PADRONIZADOS")
    logger.info("=" * 80)

    df_ols = df[[col_alvo] + regressores].dropna()
    if len(df_ols) < 20:
        return

    # Padroniza para comparar magnitude dos coeficientes
    df_std = (df_ols - df_ols.mean()) / df_ols.std()
    X = sm.add_constant(df_std[regressores])
    y = df_std[col_alvo]
    modelo = sm.OLS(y, X).fit()

    coefs  = modelo.params.drop('const')
    erros  = modelo.bse.drop('const')
    pvalues = modelo.pvalues.drop('const')

    ordem = coefs.abs().sort_values(ascending=True).index
    coefs   = coefs[ordem]
    erros   = erros[ordem]
    pvalues = pvalues[ordem]

    cores_bar = ['#e74c3c' if v > 0 else '#2980b9' for v in coefs.values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(coefs) * 0.7 + 1)))
    fig.suptitle(
        f'Regressao OLS — Coeficientes Padronizados\n'
        f'Variavel dependente: {col_alvo} | R²={modelo.rsquared:.3f}',
        fontsize=12, fontweight='bold'
    )

    ax.barh(range(len(coefs)), coefs.values, color=cores_bar, alpha=0.8,
            xerr=1.96 * erros.values, capsize=4, error_kw={'lw': 1.5})
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs.index, fontsize=9)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Coeficiente Beta Padronizado')
    ax.grid(True, axis='x', alpha=0.3)

    for i, (v, p) in enumerate(zip(coefs.values, pvalues.values)):
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        offset = erros.values[i] * 1.96 + 0.02
        ax.text(v + offset if v >= 0 else v - offset,
                i, sig, va='center', ha='left' if v >= 0 else 'right', fontsize=9)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_08_coeficientes_ols.png',
                             'OLS - Coeficientes Beta Padronizados')
    logger.info(f"   [OK] OLS gerado. R²={modelo.rsquared:.3f}")

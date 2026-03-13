import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .config import LIMIARES
from .constants import MESES_PT

logger = logging.getLogger(__name__)


def gerar_graficos_contexto_macro(df_desemprego, df_ipca_ibge, df_inadt, saida_graficos):
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
        saida_graficos.registrar(fig, '02_ipca_variacao_mensal.png',
                                 'IPCA - Variacao Mensal (%)')
        logger.info("   [OK] IPCA mensal IBGE gerado.")

    # Painel comparativo: Desemprego + IPCA mensal + Inadimplencia Total
    _painel_itens = [
        (df_desemprego, 'desemprego',              '#e67e22', 'Desocupacao (%)'),
        (df_ipca_ibge,  'ipca_mensal',             '#e74c3c', 'IPCA Mensal (%)'),
        (df_inadt,      'tx_inadimplencia_total',  '#2980b9', 'Inadimplencia Total (%)'),
    ]
    _painel_validos = [(d, c, cor, lbl) for d, c, cor, lbl in _painel_itens
                       if not d.empty and c in d.columns]

    if len(_painel_validos) >= 2:
        fig, axes = plt.subplots(len(_painel_validos), 1,
                                 figsize=(16, 3.5 * len(_painel_validos)),
                                 sharex=True)
        fig.suptitle('Contexto Macroeconomico: IBGE + BACEN', fontsize=13, fontweight='bold')
        axes = np.atleast_1d(axes).ravel()

        for idx, (df_src, col, cor, lbl) in enumerate(_painel_validos):
            axes[idx].plot(df_src['data'], df_src[col], lw=2, color=cor, label=lbl)
            axes[idx].fill_between(df_src['data'], df_src[col], alpha=0.12, color=cor)
            axes[idx].set_ylabel(lbl, fontsize=9)
            axes[idx].legend(fontsize=8, loc='upper left')
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Data')
        plt.tight_layout()
        saida_graficos.registrar(fig, '03_contexto_macroeconomico.png',
                                 'Contexto Macroeconomico - Desemprego, IPCA e Inadimplencia')
        logger.info("   [OK] Painel comparativo macro gerado.")


def gerar_boxplots_outliers(df, num_cols, saida_graficos):
    if num_cols:
        n_cols_viz = min(len(num_cols), 6)
        cols_viz = num_cols[:n_cols_viz]
        n_rows = (n_cols_viz + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4 + 1))
        fig.suptitle('Boxplots - Deteccao de Outliers', fontsize=14, fontweight='bold')
        axes = np.atleast_1d(axes).ravel()

        for idx, col in enumerate(cols_viz):
            serie = df[col].dropna()
            axes[idx].boxplot(serie, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='#4a90d9', alpha=0.7))
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_ylabel('Valor')

        for ax in axes[len(cols_viz):]:
            ax.set_visible(False)

        plt.tight_layout()
        saida_graficos.registrar(fig, '04_outliers_boxplots.png',
                                 'Outliers - Analise por Boxplot')


def gerar_distribuicoes_numericas(df, num_cols, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("5) DISTRIBUICAO DAS VARIAVEIS NUMERICAS")
    logger.info("=" * 80)

    if num_cols:
        n_plot = min(len(num_cols), 9)
        n_r = (n_plot + 2) // 3
        fig, axes = plt.subplots(n_r, 3, figsize=(18, n_r * 4 + 1))
        fig.suptitle('Distribuicao das Variaveis Numericas', fontsize=14, fontweight='bold')
        axes = np.atleast_1d(axes).ravel()

        for idx, col in enumerate(num_cols[:n_plot]):
            serie = df[col].dropna()
            sns.histplot(serie, kde=True, ax=axes[idx], bins=30, color='steelblue')
            axes[idx].axvline(serie.mean(), color='red', ls='--', lw=1.5,
                              label=f'Media: {serie.mean():.2f}')
            axes[idx].axvline(serie.median(), color='green', ls=':', lw=1.5,
                              label=f'Mediana: {serie.median():.2f}')
            axes[idx].set_title(col, fontsize=9)
            axes[idx].legend(fontsize=7)

        for ax in axes[n_plot:]:
            ax.set_visible(False)

        plt.tight_layout()
        saida_graficos.registrar(fig, '05_distribuicao_variaveis_numericas.png',
                                 'Distribuicao das Variaveis Numericas')
        logger.info("   [OK] Histogramas gerados.")


def gerar_distribuicoes_categoricas(df, cat_cols, usando_macro, saida_graficos):
    if not usando_macro and cat_cols:
        logger.info("\n" + "=" * 80)
        logger.info("6) DISTRIBUICAO DAS VARIAVEIS CATEGORICAS")
        logger.info("=" * 80)

        fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 6))
        fig.suptitle('Distribuicao das Variaveis Categoricas (Top 10)',
                     fontsize=14, fontweight='bold')
        axes = np.atleast_1d(axes).ravel()

        for idx, col in enumerate(cat_cols):
            top_vals = df[col].value_counts().head(10)
            axes[idx].barh(top_vals.index[::-1], top_vals.values[::-1],
                           color=sns.color_palette('husl', 10))
            axes[idx].set_title(col, fontsize=11)
            axes[idx].set_xlabel('Quantidade')
            for i, v in enumerate(top_vals.values[::-1]):
                axes[idx].text(v + 0.5, i, f'{v:,}', va='center', fontsize=7)

        plt.tight_layout()
        saida_graficos.registrar(fig, '06_distribuicao_variaveis_categoricas.png',
                                 'Distribuicao das Variaveis Categoricas')
        logger.info("   [OK] Graficos de categorias gerados.")


def gerar_graficos_variavel_alvo(df, col_alvo, usando_macro, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("7) VARIAVEL ALVO: TAXA DE INADIMPLENCIA")
    logger.info("=" * 80)

    if col_alvo:
        serie_alvo = df[col_alvo].dropna()
        q25, mediana, q75 = serie_alvo.quantile([0.25, 0.5, 0.75])

        logger.info(f"\n   Coluna alvo : {col_alvo}")
        logger.info(f"   N           : {len(serie_alvo):,}")
        logger.info(f"   Media       : {serie_alvo.mean():.3f}%")
        logger.info(f"   Mediana     : {mediana:.3f}%")
        logger.info(f"   Std         : {serie_alvo.std():.3f}%")
        logger.info(f"   Min/Max     : {serie_alvo.min():.3f}% / {serie_alvo.max():.3f}%")
        logger.info(f"   Q1/Q3       : {q25:.3f}% / {q75:.3f}%")

        bins = [-np.inf, q25, mediana, q75, np.inf]
        labels = ['Baixa', 'Media-Baixa', 'Media-Alta', 'Alta']
        df['nivel_inadimplencia'] = pd.cut(df[col_alvo], bins=bins, labels=labels)

        contagens = df['nivel_inadimplencia'].value_counts().reindex(labels)
        logger.info("\n   Distribuicao por nivel:")
        for nivel, cnt in contagens.items():
            if pd.notna(cnt):
                logger.info(f"      {nivel:<15}: {int(cnt):>6} ({int(cnt)/len(df)*100:.1f}%)")

        cores_nivel = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Distribuicao da Variavel Alvo: Taxa de Inadimplencia',
                     fontsize=13, fontweight='bold')

        sns.histplot(serie_alvo, kde=True, ax=axes[0], bins=35, color='#4a90d9')
        axes[0].axvline(serie_alvo.mean(), color='red', ls='--',
                        label=f'Media {serie_alvo.mean():.2f}%')
        axes[0].axvline(mediana, color='green', ls=':',
                        label=f'Mediana {mediana:.2f}%')
        axes[0].set_title('Distribuicao da Taxa de Inadimplencia')
        axes[0].set_xlabel('Taxa (%)')
        axes[0].legend(fontsize=8)

        cnt_vals = [contagens.get(lbl, 0) for lbl in labels]
        bars = axes[1].bar(labels, cnt_vals, color=cores_nivel, edgecolor='black')
        axes[1].set_title('Distribuicao por Nivel')
        axes[1].set_ylabel('Quantidade de registros')
        for bar, v in zip(bars, cnt_vals):
            if v > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                             f'{int(v)}\n({int(v)/len(df)*100:.1f}%)',
                             ha='center', va='bottom', fontsize=8)

        cnt_valid    = [v for v in cnt_vals if v > 0]
        labels_valid = [lbl for lbl, v in zip(labels, cnt_vals) if v > 0]
        cores_valid  = [c for c, v in zip(cores_nivel, cnt_vals) if v > 0]
        axes[2].pie(cnt_valid, labels=labels_valid, autopct='%1.1f%%',
                    colors=cores_valid, startangle=90)
        axes[2].set_title('Proporcao por Nivel')

        plt.tight_layout()
        saida_graficos.registrar(fig, '07_distribuicao_taxa_inadimplencia.png',
                                 'Distribuicao da Taxa de Inadimplencia')
        logger.info("   [OK] Graficos da variavel alvo gerados.")
    else:
        logger.warning("   [AVISO] Coluna de inadimplencia nao encontrada.")


def gerar_analises_bivariadas(df, col_alvo, num_cols, cat_cols, usando_macro, saida_graficos):
    if col_alvo and num_cols:
        logger.info("\n" + "=" * 80)
        logger.info("8) ANALISE BIVARIADA: INADIMPLENCIA vs NUMERICAS")
        logger.info("=" * 80)

        outras_num = [c for c in num_cols if c != col_alvo][:4]
        if outras_num:
            n_c = len(outras_num)
            fig, axes = plt.subplots(1, n_c, figsize=(5 * n_c, 5))
            fig.suptitle('Taxa de Inadimplencia vs Variaveis Numericas',
                         fontsize=13, fontweight='bold')
            axes = np.atleast_1d(axes).ravel()

            for idx, col in enumerate(outras_num):
                df_s = df[[col_alvo, col]].dropna()
                if len(df_s) < 10:
                    logger.warning(
                        f"   [AVISO] {col}: apenas {len(df_s)} amostras - scatter omitido."
                    )
                    axes[idx].set_visible(False)
                    continue
                axes[idx].scatter(df_s[col], df_s[col_alvo],
                                  alpha=0.4, s=20, color='steelblue')
                m, b, r, p, _ = stats.linregress(df_s[col], df_s[col_alvo])
                x_range = np.linspace(df_s[col].min(), df_s[col].max(), 100)
                axes[idx].plot(x_range, m * x_range + b, 'r-', lw=2,
                               label=f'r={r:.2f} (p={p:.3f})')
                axes[idx].set_xlabel(col, fontsize=9)
                axes[idx].set_ylabel(col_alvo, fontsize=9)
                axes[idx].set_title(f'{col_alvo} vs {col}', fontsize=9)
                axes[idx].legend(fontsize=7)

            plt.tight_layout()
            saida_graficos.registrar(fig, '08_inadimplencia_vs_numericas.png',
                                     'Inadimplencia vs Variaveis Numericas')
            logger.info("   [OK] Scatter plots bivariados gerados.")

    if col_alvo and not usando_macro and cat_cols:
        logger.info("\n" + "=" * 80)
        logger.info("9) ANALISE BIVARIADA: INADIMPLENCIA vs CATEGORICAS")
        logger.info("=" * 80)

        cols_cat_analise = [c for c in cat_cols if c != 'InstituicaoFinanceira']
        n_cat = len(cols_cat_analise)
        if n_cat > 0:
            fig, axes = plt.subplots(1, n_cat, figsize=(8 * n_cat, 6))
            fig.suptitle(f'{col_alvo} por Variaveis Categoricas',
                         fontsize=13, fontweight='bold')
            axes = np.atleast_1d(axes).ravel()

            for idx, col in enumerate(cols_cat_analise):
                medias = (df.groupby(col)[col_alvo]
                            .mean()
                            .sort_values(ascending=False)
                            .head(15))
                # RdYlGn_r: verde(baixo) -> amarelo -> vermelho(alto)
                # barh com valores revertidos: bottom=menor, top=maior inadimplencia
                cores = sns.color_palette('RdYlGn_r', len(medias))
                axes[idx].barh(medias.index[::-1], medias.values[::-1], color=cores)
                axes[idx].set_title(f'Media de {col_alvo} por {col}', fontsize=9)
                axes[idx].set_xlabel('Taxa Media (%)')
                for i, v in enumerate(medias.values[::-1]):
                    axes[idx].text(v, i, f' {v:.2f}%', va='center', fontsize=7)

            plt.tight_layout()
            saida_graficos.registrar(fig, '09_inadimplencia_por_modalidade.png',
                                     'Inadimplencia por Modalidade e Segmento')
            logger.info("   [OK] Graficos por categoria gerados.")


def gerar_analise_temporal(df, df_macro, col_alvo, usando_macro, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("10) ANALISE TEMPORAL")
    logger.info("=" * 80)

    if not df_macro.empty and 'data' in df_macro.columns:
        df_t = df_macro.copy()
        df_t['ano'] = df_t['data'].dt.year
        df_t['mes'] = df_t['data'].dt.month

        cols_macro_disp = [c for c in ['tx_inadimplencia_total', 'tx_inadimplencia_pf',
                                        'selic_aa', 'ipca_acum12m', 'desemprego']
                           if c in df_t.columns]

        if cols_macro_disp:
            n_c = len(cols_macro_disp)
            fig, axes = plt.subplots(n_c, 1, figsize=(16, 3.5 * n_c))
            fig.suptitle('Series Temporais - Indicadores Macroeconomicos (BACEN/IBGE)',
                         fontsize=13, fontweight='bold')
            axes = np.atleast_1d(axes).ravel()

            labels_map = {
                'tx_inadimplencia_total': 'Inadimplencia Total (%)',
                'tx_inadimplencia_pf':    'Inadimplencia PF (%)',
                'selic_aa':               'Selic (% a.a.)',
                'ipca_acum12m':           'IPCA acum. 12m (%)',
                'desemprego':             'Desocupacao PNAD (%)',
            }
            cores_ts = ['#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#27ae60']

            for idx, col in enumerate(cols_macro_disp):
                df_col = df_t[['data', col]].dropna()
                axes[idx].plot(df_col['data'], df_col[col],
                               lw=2, color=cores_ts[idx % len(cores_ts)])
                mm12 = df_col[col].rolling(12, min_periods=3).mean()
                axes[idx].plot(df_col['data'], mm12, lw=1.5, ls='--',
                               color='black', alpha=0.6, label='MM 12m')
                axes[idx].set_ylabel(labels_map.get(col, col), fontsize=9)
                axes[idx].legend(fontsize=8)
                axes[idx].grid(True, alpha=0.3)

            axes[-1].set_xlabel('Data')
            plt.tight_layout()
            saida_graficos.registrar(fig, '10_series_temporais_macro.png',
                                     'Series Temporais - Indicadores Macro BACEN')
            logger.info("   [OK] Series temporais geradas.")

        if not usando_macro and 'Mes' in df.columns and col_alvo in df.columns:
            df_temp = (df.groupby('Mes')[col_alvo]
                         .mean()
                         .reset_index()
                         .sort_values('Mes'))

            fig = plt.figure(figsize=(14, 5))
            plt.plot(df_temp['Mes'], df_temp[col_alvo], lw=2, color='#e74c3c')
            mm6 = df_temp[col_alvo].rolling(6, min_periods=2).mean()
            plt.plot(df_temp['Mes'], mm6, lw=2, ls='--', color='black',
                     alpha=0.7, label='MM 6m')
            plt.title(f'Evolucao Media Mensal - {col_alvo} (por modalidade)',
                      fontweight='bold')
            plt.xlabel('Mes')
            plt.ylabel('Taxa Media (%)')
            plt.legend()
            plt.tight_layout()
            saida_graficos.registrar(fig, '11_evolucao_mensal_inadimplencia.png',
                                     'Evolucao Mensal da Taxa de Inadimplencia')
            logger.info("   [OK] Evolucao mensal gerada.")


def gerar_matriz_correlacao(df, num_cols, df_macro, usando_macro, col_alvo, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("11) MATRIZ DE CORRELACAO")
    logger.info("=" * 80)

    if not usando_macro and num_cols and len(num_cols) >= 2:
        df_corr_data = df[num_cols].dropna()
    elif not df_macro.empty:
        cols_macro_num = [c for c in df_macro.columns if c != 'data']
        df_corr_data = df_macro[cols_macro_num].dropna()
        num_cols = cols_macro_num
    else:
        df_corr_data = pd.DataFrame()

    if not df_corr_data.empty and df_corr_data.shape[1] >= 2:
        corr_matrix = df_corr_data.corr()

        fig = plt.figure(figsize=(max(8, len(corr_matrix)), max(6, len(corr_matrix) - 1)))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, square=True,
                    linewidths=0.5, vmin=-1, vmax=1,
                    cbar_kws={'shrink': 0.8})
        plt.title('Matriz de Correlacao', fontsize=13, fontweight='bold')
        plt.tight_layout()
        saida_graficos.registrar(fig, '12_matriz_correlacao.png', 'Matriz de Correlacao')
        logger.info("   [OK] Matriz de correlacao gerada.")

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


def gerar_analise_avancada(df, col_alvo, df_macro, usando_macro, saida_graficos):
    logger.info("\n" + "=" * 80)
    logger.info("12) ANALISE AVANCADA")
    logger.info("=" * 80)

    if not usando_macro and not df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        titulo_fig = 'Analise Avancada - Inadimplencia por Modalidade'
        fig.suptitle(titulo_fig, fontsize=13, fontweight='bold')

        if 'Modalidade' in df.columns and col_alvo in df.columns:
            top_mod = (df.groupby('Modalidade')[col_alvo]
                         .mean()
                         .sort_values(ascending=False)
                         .head(15))
            cores_mod = sns.color_palette('RdYlGn_r', len(top_mod))
            axes[0, 0].barh(top_mod.index[::-1], top_mod.values[::-1], color=cores_mod)
            axes[0, 0].set_title('Top 15 Modalidades - Maior Inadimplencia Media')
            axes[0, 0].set_xlabel('Taxa Media (%)')
            for i, v in enumerate(top_mod.values[::-1]):
                axes[0, 0].text(v, i, f' {v:.2f}%', va='center', fontsize=7)

        if 'Segmento' in df.columns and col_alvo in df.columns:
            segmentos = df['Segmento'].value_counts().head(6).index.tolist()
            df_seg = df[df['Segmento'].isin(segmentos)]
            df_seg.boxplot(column=col_alvo, by='Segmento', ax=axes[0, 1])
            axes[0, 1].set_title('Distribuicao por Segmento')
            axes[0, 1].set_xlabel('Segmento')
            axes[0, 1].set_ylabel('Taxa (%)')
            plt.sca(axes[0, 1])
            plt.xticks(rotation=30, ha='right')
            # Restaura suptitle sobrescrito pelo boxplot
            fig.suptitle(titulo_fig, fontsize=13, fontweight='bold')

        if ('TaxaJurosAoMes' in df.columns and col_alvo in df.columns
                and 'Segmento' in df.columns):
            df_sc = df[['TaxaJurosAoMes', col_alvo, 'Segmento']].dropna()
            segmentos_uniq = df_sc['Segmento'].unique()[:6]
            cores_seg = sns.color_palette('husl', len(segmentos_uniq))
            for i, seg in enumerate(segmentos_uniq):
                mask = df_sc['Segmento'] == seg
                axes[1, 0].scatter(df_sc.loc[mask, 'TaxaJurosAoMes'],
                                   df_sc.loc[mask, col_alvo],
                                   alpha=0.4, s=15, color=cores_seg[i], label=seg)
            axes[1, 0].set_xlabel('Taxa de Juros ao Mes (%)')
            axes[1, 0].set_ylabel(f'{col_alvo} (%)')
            axes[1, 0].set_title('Taxa de Juros vs Inadimplencia')
            axes[1, 0].legend(fontsize=7, ncol=2)

        if 'Mes' in df.columns and 'Segmento' in df.columns and col_alvo in df.columns:
            df['_ano'] = pd.to_datetime(df['Mes'], errors='coerce').dt.year
            pivot = (df.groupby(['_ano', 'Segmento'])[col_alvo]
                       .mean()
                       .unstack('Segmento'))
            pivot = pivot.dropna(axis=1, how='all').iloc[-8:]
            df.drop(columns='_ano', inplace=True)
            if not pivot.empty and pivot.shape[1] >= 2:
                sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                            ax=axes[1, 1], linewidths=0.5)
                axes[1, 1].set_title('Heatmap: Inadimplencia Media (Ano x Segmento)')
                axes[1, 1].set_xlabel('Segmento')
                axes[1, 1].set_ylabel('Ano')
            else:
                axes[1, 1].set_visible(False)
        else:
            axes[1, 1].set_visible(False)

        plt.tight_layout()
        saida_graficos.registrar(fig, '13_analise_avancada.png', 'Analise Avancada')
        logger.info("   [OK] Analise avancada gerada.")

    elif not df_macro.empty and len(df_macro) > 12:
        df_t2 = df_macro.copy()
        df_t2['ano'] = df_t2['data'].dt.year
        df_t2['mes'] = df_t2['data'].dt.month

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Analise Avancada - Indicadores Macro', fontsize=13, fontweight='bold')

        if 'tx_inadimplencia_total' in df_t2.columns:
            media_mes = df_t2.groupby('mes')['tx_inadimplencia_total'].mean()
            axes[0, 0].bar(range(1, 13), media_mes.reindex(range(1, 13)),
                           color='steelblue', edgecolor='black')
            axes[0, 0].set_title('Sazonalidade - Media por Mes')
            axes[0, 0].set_xticks(range(1, 13))
            axes[0, 0].set_xticklabels(MESES_PT, rotation=45)
            axes[0, 0].set_ylabel('Taxa Media (%)')

        if 'tx_inadimplencia_total' in df_t2.columns:
            media_ano = df_t2.groupby('ano')['tx_inadimplencia_total'].mean()
            bars = axes[0, 1].bar(media_ano.index, media_ano.values,
                                  color='coral', edgecolor='black')
            axes[0, 1].set_title('Media Anual - Inadimplencia Total')
            axes[0, 1].set_xlabel('Ano')
            axes[0, 1].set_ylabel('Taxa Media (%)')
            axes[0, 1].set_xticks(media_ano.index)
            axes[0, 1].set_xticklabels(media_ano.index.astype(str), rotation=45)
            for bar in bars:
                h = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, h,
                                f'{h:.2f}%', ha='center', va='bottom', fontsize=8)

        if 'selic_aa' in df_t2.columns and 'tx_inadimplencia_total' in df_t2.columns:
            df_sc2 = df_t2[['selic_aa', 'tx_inadimplencia_total']].dropna()
            axes[1, 0].scatter(df_sc2['selic_aa'], df_sc2['tx_inadimplencia_total'],
                               alpha=0.6, s=30, color='#8e44ad')
            m, b, r, p, _ = stats.linregress(df_sc2['selic_aa'],
                                              df_sc2['tx_inadimplencia_total'])
            xr = np.linspace(df_sc2['selic_aa'].min(), df_sc2['selic_aa'].max(), 100)
            axes[1, 0].plot(xr, m * xr + b, 'r-', lw=2, label=f'r={r:.2f}')
            axes[1, 0].set_xlabel('Selic (% a.a.)')
            axes[1, 0].set_ylabel('Inadimplencia Total (%)')
            axes[1, 0].set_title('Selic vs Inadimplencia')
            axes[1, 0].legend()

        if 'ipca_acum12m' in df_t2.columns and 'tx_inadimplencia_total' in df_t2.columns:
            df_sc3 = df_t2[['ipca_acum12m', 'tx_inadimplencia_total']].dropna()
            axes[1, 1].scatter(df_sc3['ipca_acum12m'], df_sc3['tx_inadimplencia_total'],
                               alpha=0.6, s=30, color='#e67e22')
            m, b, r, p, _ = stats.linregress(df_sc3['ipca_acum12m'],
                                              df_sc3['tx_inadimplencia_total'])
            xr = np.linspace(df_sc3['ipca_acum12m'].min(), df_sc3['ipca_acum12m'].max(), 100)
            axes[1, 1].plot(xr, m * xr + b, 'r-', lw=2, label=f'r={r:.2f}')
            axes[1, 1].set_xlabel('IPCA acum. 12m (%)')
            axes[1, 1].set_ylabel('Inadimplencia Total (%)')
            axes[1, 1].set_title('IPCA vs Inadimplencia')
            axes[1, 1].legend()

        plt.tight_layout()
        saida_graficos.registrar(fig, '14_analise_avancada_macro.png',
                                 'Analise Avancada - Indicadores Macro')
        logger.info("   [OK] Analise avancada macro gerada.")


def insights_finais_inadimplencia(df, col_alvo, df_macro, usando_macro):
    logger.info("\n" + "=" * 80)
    logger.info("13) INSIGHTS FINAIS E CONCLUSOES")
    logger.info("=" * 80)

    if col_alvo and col_alvo in df.columns:
        serie_alvo = df[col_alvo].dropna()
        logger.info(f"""
   VARIAVEL ALVO: {col_alvo}
   \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
   Media    : {serie_alvo.mean():.3f}%
   Mediana  : {serie_alvo.median():.3f}%
   Std      : {serie_alvo.std():.3f}%
   Min/Max  : {serie_alvo.min():.3f}% / {serie_alvo.max():.3f}%
""")

    if not usando_macro and 'Modalidade' in df.columns and col_alvo in df.columns:
        top3_alto  = df.groupby('Modalidade')[col_alvo].mean().sort_values(ascending=False).head(3)
        top3_baixo = df.groupby('Modalidade')[col_alvo].mean().sort_values(ascending=True).head(3)
        logger.info("   TOP 3 MODALIDADES - MAIOR INADIMPLENCIA:")
        for mod, v in top3_alto.items():
            logger.info(f"      {mod}: {v:.2f}%")
        logger.info("\n   TOP 3 MODALIDADES - MENOR INADIMPLENCIA:")
        for mod, v in top3_baixo.items():
            logger.info(f"      {mod}: {v:.2f}%")

    if not df_macro.empty and 'tx_inadimplencia_total' in df_macro.columns:
        serie_t    = df_macro['tx_inadimplencia_total'].dropna()
        media_anual = df_macro.groupby(df_macro['data'].dt.year)['tx_inadimplencia_total'].mean()
        ano_max = media_anual.idxmax()
        ano_min = media_anual.idxmin()
        logger.info(f"""
   SERIE MACRO (BACEN SGS 21082):
   \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
   Ano maior inadimplencia: {ano_max} ({media_anual[ano_max]:.2f}%)
   Ano menor inadimplencia: {ano_min} ({media_anual[ano_min]:.2f}%)
   Amplitude total        : {serie_t.max() - serie_t.min():.2f} p.p.
""")

    logger.info("""
   PRINCIPAIS ACHADOS:
   \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
   1. FONTES: Dados 100% publicos e verificaveis (BACEN + IBGE).
      Nenhum dado simulado utilizado nesta analise.

   2. GRANULARIDADE: Dataset Olinda oferece visao por modalidade de
      credito (cartao, consignado, pessoal etc.) e segmento (PF/PJ),
      permitindo identificar os produtos de maior risco.

   3. MACRO-CONTEXTO: Selic, IPCA e desemprego explicam parte da
      variabilidade da inadimplencia ao longo do tempo. Periodos de
      juros altos e desemprego elevado tendem a elevar a inadimplencia.

   4. MODALIDADES DE RISCO: Credito pessoal nao consignado e cartao
      rotativo historicamente apresentam as maiores taxas de
      inadimplencia no mercado brasileiro.

   5. LIMITACAO: Dados de clientes individuais nao estao disponiveis
      em APIs publicas (LGPD). Para modelagem preditiva seria
      necessario dataset proprietario de uma instituicao financeira.
""")


def gerar_ranking_instituicoes(df, col_alvo, saida_graficos):
    if col_alvo not in df.columns or 'InstituicaoFinanceira' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) RANKING DE INSTITUICOES FINANCEIRAS")
    logger.info("=" * 80)

    medias = df.groupby('InstituicaoFinanceira')[col_alvo].mean()
    top10  = medias.sort_values(ascending=False).head(10)
    bot10  = medias.sort_values(ascending=True).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Ranking de Instituicoes — Taxa Media de Inadimplencia',
                 fontsize=13, fontweight='bold')

    cores_top = sns.color_palette('Reds_r', len(top10))
    axes[0].barh(top10.index[::-1], top10.values[::-1], color=cores_top)
    axes[0].set_title('Top 10 — Maior Inadimplencia')
    axes[0].set_xlabel('Taxa Media (%)')
    for i, v in enumerate(top10.values[::-1]):
        axes[0].text(v, i, f' {v:.2f}%', va='center', fontsize=8)

    cores_bot = sns.color_palette('Greens', len(bot10))
    axes[1].barh(bot10.index, bot10.values, color=cores_bot)
    axes[1].set_title('Top 10 — Menor Inadimplencia')
    axes[1].set_xlabel('Taxa Media (%)')
    for i, v in enumerate(bot10.values):
        axes[1].text(v, i, f' {v:.2f}%', va='center', fontsize=8)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_01_ranking_instituicoes.png',
                             'Ranking de Instituicoes - Inadimplencia Media')
    logger.info("   [OK] Ranking de instituicoes gerado.")


def gerar_bubble_chart_modalidades(df, col_alvo, saida_graficos):
    cols_req = [col_alvo, 'TaxaJurosAoMes', 'NumeroDeContratos', 'Modalidade']
    if not all(c in df.columns for c in cols_req):
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) BUBBLE CHART: JUROS x INADIMPLENCIA x VOLUME")
    logger.info("=" * 80)

    agg = df.groupby('Modalidade').agg(
        inadimplencia=(col_alvo, 'mean'),
        juros=('TaxaJurosAoMes', 'mean'),
        contratos=('NumeroDeContratos', 'sum')
    ).dropna().reset_index()

    agg = agg[agg['contratos'] > 0]
    tamanhos = (agg['contratos'] / agg['contratos'].max() * 1200) + 80
    cores_b  = sns.color_palette('husl', len(agg))

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Taxa de Juros vs Inadimplencia vs Volume de Contratos (por Modalidade)',
                 fontsize=12, fontweight='bold')

    scatter = ax.scatter(agg['juros'], agg['inadimplencia'],
                         s=tamanhos, c=range(len(agg)),
                         cmap='husl', alpha=0.7, edgecolors='black', lw=0.5)

    for _, row in agg.iterrows():
        ax.annotate(row['Modalidade'],
                    (row['juros'], row['inadimplencia']),
                    fontsize=6.5, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points')

    ax.set_xlabel('Taxa de Juros ao Mes (% media)')
    ax.set_ylabel('Taxa de Inadimplencia (% media)')
    ax.grid(True, alpha=0.3)

    import matplotlib.lines as mlines
    for sz, lbl in [(80, 'Baixo'), (500, 'Medio'), (1200, 'Alto')]:
        ax.scatter([], [], s=sz, color='gray', alpha=0.5, label=f'Volume: {lbl}')
    ax.legend(title='Volume de Contratos', fontsize=8, loc='upper left')

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_02_bubble_chart_modalidades.png',
                             'Bubble Chart - Juros vs Inadimplencia vs Volume')
    logger.info("   [OK] Bubble chart gerado.")


def gerar_evolucao_top_modalidades(df, col_alvo, saida_graficos):
    if col_alvo not in df.columns or 'Modalidade' not in df.columns or 'Mes' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) EVOLUCAO TEMPORAL DAS TOP MODALIDADES")
    logger.info("=" * 80)

    top5 = (df.groupby('Modalidade')[col_alvo]
              .mean()
              .sort_values(ascending=False)
              .head(5)
              .index.tolist())

    df_t = df[df['Modalidade'].isin(top5)].copy()
    df_t['Mes'] = pd.to_datetime(df_t['Mes'], errors='coerce')
    df_t = df_t.dropna(subset=['Mes'])
    evolucao = (df_t.groupby(['Mes', 'Modalidade'])[col_alvo]
                    .mean()
                    .reset_index())

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle('Evolucao da Inadimplencia — Top 5 Modalidades',
                 fontsize=13, fontweight='bold')

    cores_ev = sns.color_palette('husl', len(top5))
    for i, mod in enumerate(top5):
        df_mod = evolucao[evolucao['Modalidade'] == mod].sort_values('Mes')
        ax.plot(df_mod['Mes'], df_mod[col_alvo],
                lw=2, color=cores_ev[i], label=mod, marker='o', ms=3)

    ax.set_xlabel('Data')
    ax.set_ylabel('Taxa de Inadimplencia (%)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_03_evolucao_top_modalidades.png',
                             'Evolucao das Top 5 Modalidades por Inadimplencia')
    logger.info("   [OK] Evolucao top modalidades gerada.")


def gerar_violin_por_segmento(df, col_alvo, saida_graficos):
    if col_alvo not in df.columns or 'Segmento' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) VIOLIN PLOT POR SEGMENTO")
    logger.info("=" * 80)

    df_v = df[['Segmento', col_alvo]].dropna()
    segmentos = df_v['Segmento'].value_counts().head(6).index.tolist()
    df_v = df_v[df_v['Segmento'].isin(segmentos)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Distribuicao de {col_alvo} por Segmento',
                 fontsize=13, fontweight='bold')

    sns.violinplot(data=df_v, x='Segmento', y=col_alvo,
                   palette='husl', inner='box', ax=axes[0])
    axes[0].set_title('Violin Plot')
    axes[0].set_xlabel('Segmento')
    axes[0].set_ylabel('Taxa (%)')
    axes[0].tick_params(axis='x', rotation=30)

    sns.boxplot(data=df_v, x='Segmento', y=col_alvo,
                palette='husl', ax=axes[1])
    axes[1].set_title('Boxplot Comparativo')
    axes[1].set_xlabel('Segmento')
    axes[1].set_ylabel('Taxa (%)')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_04_violin_por_segmento.png',
                             'Violin e Boxplot por Segmento')
    logger.info("   [OK] Violin por segmento gerado.")


def gerar_heatmap_instituicao_modalidade(df, col_alvo, saida_graficos):
    if (col_alvo not in df.columns
            or 'InstituicaoFinanceira' not in df.columns
            or 'Modalidade' not in df.columns):
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) HEATMAP INSTITUICAO x MODALIDADE")
    logger.info("=" * 80)

    top_inst = df['InstituicaoFinanceira'].value_counts().head(15).index
    top_mod  = df['Modalidade'].value_counts().head(10).index
    df_h = df[df['InstituicaoFinanceira'].isin(top_inst) &
               df['Modalidade'].isin(top_mod)]

    pivot = df_h.pivot_table(
        values=col_alvo,
        index='InstituicaoFinanceira',
        columns='Modalidade',
        aggfunc='mean'
    )
    pivot = pivot.dropna(how='all').fillna(0)

    fig = plt.figure(figsize=(max(12, pivot.shape[1] * 1.4),
                               max(8, pivot.shape[0] * 0.5 + 2)))
    fig.suptitle('Inadimplencia Media — Top Instituicoes x Top Modalidades',
                 fontsize=12, fontweight='bold')

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.4, cbar_kws={'shrink': 0.7},
                annot_kws={'size': 7})
    plt.xticks(rotation=40, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_05_heatmap_inst_modalidade.png',
                             'Heatmap Instituicao x Modalidade')
    logger.info("   [OK] Heatmap instituicao x modalidade gerado.")


def gerar_concentracao_mercado(df, saida_graficos):
    col_vol = 'NumeroDeContratos' if 'NumeroDeContratos' in df.columns else (
              'BaseDeCalculo'     if 'BaseDeCalculo'     in df.columns else None)
    if col_vol is None or 'InstituicaoFinanceira' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) CONCENTRACAO DE MERCADO (PARETO)")
    logger.info("=" * 80)

    participacao = (df.groupby('InstituicaoFinanceira')[col_vol]
                      .sum()
                      .sort_values(ascending=False))
    participacao_pct = (participacao / participacao.sum() * 100).head(20)
    acumulado = participacao_pct.cumsum()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.suptitle(f'Concentracao de Mercado por Instituicao ({col_vol})',
                 fontsize=13, fontweight='bold')

    ax2 = ax1.twinx()
    cores_p = ['#e74c3c' if v >= 80 else '#f1c40f' if v >= 50 else '#27ae60'
               for v in acumulado.values]
    ax1.bar(range(len(participacao_pct)), participacao_pct.values,
            color=cores_p, alpha=0.8, edgecolor='black', lw=0.5)
    ax2.plot(range(len(acumulado)), acumulado.values,
             color='#2c3e50', lw=2.5, marker='o', ms=5, label='% Acumulado')
    ax2.axhline(80, color='gray', ls='--', lw=1, alpha=0.7, label='80%')

    ax1.set_xticks(range(len(participacao_pct)))
    ax1.set_xticklabels(participacao_pct.index, rotation=45, ha='right', fontsize=7)
    ax1.set_ylabel('Participacao (%)')
    ax2.set_ylabel('% Acumulado')
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9, loc='center right')
    ax1.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_06_concentracao_mercado.png',
                             'Concentracao de Mercado - Curva de Pareto')
    logger.info("   [OK] Concentracao de mercado gerada.")


def gerar_quartis_modalidade_temporal(df, col_alvo, saida_graficos):
    if col_alvo not in df.columns or 'Mes' not in df.columns or 'Modalidade' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) QUARTIS POR MODALIDADE AO LONGO DO TEMPO")
    logger.info("=" * 80)

    top6 = (df.groupby('Modalidade')[col_alvo]
              .mean()
              .sort_values(ascending=False)
              .head(6)
              .index.tolist())

    df_q = df[df['Modalidade'].isin(top6)].copy()
    df_q['Mes'] = pd.to_datetime(df_q['Mes'], errors='coerce')
    df_q = df_q.dropna(subset=['Mes'])

    n_mod = len(top6)
    n_cols_grid = 2
    n_rows_grid = (n_mod + 1) // n_cols_grid

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                              figsize=(16, n_rows_grid * 4))
    fig.suptitle('Variacao Quartil da Inadimplencia por Modalidade ao Longo do Tempo',
                 fontsize=12, fontweight='bold')
    axes = np.atleast_1d(axes).ravel()

    for idx, mod in enumerate(top6):
        df_mod = (df_q[df_q['Modalidade'] == mod]
                    .groupby('Mes')[col_alvo]
                    .quantile([0.25, 0.5, 0.75])
                    .unstack()
                    .sort_index())
        if df_mod.empty:
            axes[idx].set_visible(False)
            continue
        axes[idx].fill_between(df_mod.index, df_mod[0.25], df_mod[0.75],
                                alpha=0.3, color='#3498db', label='IQR (Q1-Q3)')
        axes[idx].plot(df_mod.index, df_mod[0.5],
                       lw=2, color='#2980b9', label='Mediana')
        axes[idx].set_title(mod, fontsize=9)
        axes[idx].set_ylabel('Taxa (%)')
        axes[idx].legend(fontsize=7)
        axes[idx].grid(True, alpha=0.3)

    for ax in axes[n_mod:]:
        ax.set_visible(False)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_07_quartis_modalidade_temporal.png',
                             'Quartis de Inadimplencia por Modalidade (Temporal)')
    logger.info("   [OK] Quartis por modalidade temporal gerado.")


def gerar_juros_por_segmento(df, saida_graficos):
    col_j = 'TaxaJurosAoAno' if 'TaxaJurosAoAno' in df.columns else (
            'TaxaJurosAoMes' if 'TaxaJurosAoMes' in df.columns else None)
    if col_j is None or 'Segmento' not in df.columns:
        return

    logger.info("\n" + "=" * 80)
    logger.info("EXTRA) TAXA DE JUROS POR SEGMENTO")
    logger.info("=" * 80)

    df_j = df[['Segmento', col_j]].dropna()
    segmentos = df_j['Segmento'].value_counts().head(6).index.tolist()
    df_j = df_j[df_j['Segmento'].isin(segmentos)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Taxa de Juros ({col_j}) por Segmento',
                 fontsize=13, fontweight='bold')

    sns.boxplot(data=df_j, x='Segmento', y=col_j,
                palette='Set2', ax=axes[0])
    axes[0].set_title('Boxplot por Segmento')
    axes[0].set_xlabel('Segmento')
    axes[0].set_ylabel(f'{col_j} (%)')
    axes[0].tick_params(axis='x', rotation=30)

    medias_seg = df_j.groupby('Segmento')[col_j].mean().sort_values(ascending=False)
    cores_seg  = sns.color_palette('Set2', len(medias_seg))
    axes[1].barh(medias_seg.index[::-1], medias_seg.values[::-1], color=cores_seg)
    axes[1].set_title('Media por Segmento')
    axes[1].set_xlabel(f'Media {col_j} (%)')
    for i, v in enumerate(medias_seg.values[::-1]):
        axes[1].text(v, i, f' {v:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    saida_graficos.registrar(fig, 'extra_08_juros_por_segmento.png',
                             'Taxa de Juros por Segmento')
    logger.info("   [OK] Juros por segmento gerado.")

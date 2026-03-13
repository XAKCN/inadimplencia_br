# Codigos BACEN SGS (Sistema Gerenciador de Series Temporais)
SGS = {
    'inadimplencia_total': 21082,
    'inadimplencia_pf':    21084,
    'inadimplencia_pj':    21083,
    'selic_aa':            4189,
    'ipca_acum12m':        13522,
    'concessoes_bi':       20631,
    'saldo_credito_bi':    20539,
}

# Periodo de coleta (padrao: 2015-2025)
PERIODO = {
    'data_inicial':     '01/01/2015',
    'data_final':       '31/12/2025',
    'data_inicial_iso': '2015-01-01',
    'data_final_iso':   '2025-12-31',
    'ano_inicial':      2015,
    'ano_final':        2025,
}

# Limiares estatisticos
LIMIARES = {
    'outlier_iqr':    1.5,
    'outlier_zscore': 3.0,
    'corr_forte':     0.7,
    'corr_moderada':  0.4,
}

# Saida de graficos
SAIDA = {
    'dpi_miniatura': 110,
    'dpi_painel':    150,
    'olinda_top':    50_000,
}

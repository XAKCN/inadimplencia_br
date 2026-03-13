import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def carregar_serie_bacen(codigo_sgs, data_inicial, data_final, coluna_saida, csv_saida):
    """Carrega serie temporal via BACEN SGS API (JSON) com fallback CSV local."""
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_sgs}/dados"
        f"?formato=json&dataInicial={data_inicial}&dataFinal={data_final}"
    )
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dados = resp.json()
        if not isinstance(dados, list) or len(dados) == 0:
            raise ValueError(f"Resposta vazia - SGS {codigo_sgs}")
        df = pd.DataFrame(dados)
        df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
        df['valor'] = pd.to_numeric(
            df['valor'].astype(str).str.replace(',', '.', regex=False), errors='coerce'
        )
        df.rename(columns={'valor': coluna_saida}, inplace=True)
        df = df[['data', coluna_saida]].dropna().sort_values('data').reset_index(drop=True)
        if csv_saida:
            df.to_csv(csv_saida, index=False, encoding='utf-8-sig')
        logger.info(f"   [API] SGS {codigo_sgs:>6}  ({coluna_saida}): {len(df)} obs")
        return df
    except Exception as e:
        if csv_saida and Path(csv_saida).exists():
            df = pd.read_csv(csv_saida)
            if coluna_saida not in df.columns and 'valor' in df.columns:
                df.rename(columns={'valor': coluna_saida}, inplace=True)
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            df[coluna_saida] = pd.to_numeric(df[coluna_saida], errors='coerce')
            df = df[['data', coluna_saida]].dropna().sort_values('data').reset_index(drop=True)
            logger.info(f"   [CSV] SGS {codigo_sgs:>6}  ({coluna_saida}): {len(df)} obs (cache local)")
            return df
        logger.error(f"   [ERRO] SGS {codigo_sgs}: {e}")
        return pd.DataFrame(columns=['data', coluna_saida])


def _extrair_serie_ibge(dados):
    """Extrai o dicionario {periodo: valor} da resposta padrao da API IBGE SIDRA."""
    try:
        return dados[0]['resultados'][0]['series'][0]['serie']
    except (IndexError, KeyError) as e:
        raise ValueError(f"Estrutura inesperada na resposta IBGE: {e}") from e


def carregar_desemprego_ibge(data_inicial='2015-01-01', data_final='2025-12-31',
                              csv_saida=None, mensal_interpolado=False):
    """
    Busca taxa de desocupacao trimestral (PNAD Continua) via API IBGE SIDRA.
    Se mensal_interpolado=True, aplica interpolacao linear para frequencia mensal.
    Nota: a interpolacao assume variacao linear entre trimestres - adequada para
    uso como variavel explicativa, mas nao para analise de curto prazo.
    """
    if csv_saida and Path(csv_saida).exists():
        df = pd.read_csv(csv_saida)
        df['data'] = pd.to_datetime(df['data'])
        logger.info(f"   [CSV] Desemprego IBGE: {len(df)} obs (cache local)")
        return df

    url = (
        "https://servicodados.ibge.gov.br/api/v3/agregados/4099"
        "/periodos/all/variaveis/4099?localidades=BR"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        series = _extrair_serie_ibge(resp.json())

        registros = []
        for periodo, valor in series.items():
            try:
                periodo_str = str(periodo).strip()
                if '.' in periodo_str:
                    ano_s, tri_s = periodo_str.split('.')
                elif len(periodo_str) == 5:
                    ano_s, tri_s = periodo_str[:4], periodo_str[4]
                else:
                    continue
                mes_inicio = {1: 1, 2: 4, 3: 7, 4: 10}.get(int(tri_s), 1)
                registros.append({
                    'data': pd.Timestamp(int(ano_s), mes_inicio, 1),
                    'desemprego': float(valor)
                })
            except Exception:
                continue

        if not registros:
            raise ValueError("Nenhum registro valido encontrado na resposta IBGE.")

        df_trim = pd.DataFrame(registros).sort_values('data').reset_index(drop=True)
        mask = (
            (df_trim['data'] >= pd.to_datetime(data_inicial)) &
            (df_trim['data'] <= pd.to_datetime(data_final))
        )
        df_trim = df_trim.loc[mask].reset_index(drop=True)

        if mensal_interpolado:
            df_trim = df_trim.set_index('data')
            idx_mensal = pd.date_range(df_trim.index.min(), df_trim.index.max(), freq='MS')
            df_final = df_trim.reindex(idx_mensal).interpolate(method='linear').reset_index()
            df_final.columns = ['data', 'desemprego']
        else:
            df_final = df_trim

        if csv_saida:
            df_final.to_csv(csv_saida, index=False, encoding='utf-8-sig')

        freq_str = "mensais (interpolados)" if mensal_interpolado else "obs trimestrais"
        logger.info(f"   [API] Desemprego IBGE: {len(df_final)} {freq_str}")
        return df_final
    except Exception as e:
        logger.warning(f"   [AVISO] Erro ao buscar desemprego IBGE: {e}")
        return pd.DataFrame(columns=['data', 'desemprego'])


def carregar_ipca_ibge(data_inicial='2015-01-01', data_final='2025-12-31', csv_saida=None):
    """
    Busca IPCA mensal (variacao % no mes) via API IBGE SIDRA.
    Tabela 1737 (IPCA - indice geral), variavel 2265.
    Complementa o SGS 13522 que fornece o acumulado 12 meses.
    """
    if csv_saida and Path(csv_saida).exists():
        df = pd.read_csv(csv_saida)
        df['data'] = pd.to_datetime(df['data'])
        logger.info(f"   [CSV] IPCA mensal IBGE: {len(df)} obs (cache)")
        return df

    url = (
        "https://servicodados.ibge.gov.br/api/v3/agregados/1737"
        "/periodos/all/variaveis/2265?localidades=BR"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        series = _extrair_serie_ibge(resp.json())

        df = pd.DataFrame(list(series.items()), columns=['periodo', 'ipca_mensal'])
        df['data'] = pd.to_datetime(df['periodo'], format='%Y%m', errors='coerce')
        df['ipca_mensal'] = pd.to_numeric(df['ipca_mensal'], errors='coerce')
        df = df[['data', 'ipca_mensal']].dropna().sort_values('data').reset_index(drop=True)

        mask = (
            (df['data'] >= pd.to_datetime(data_inicial)) &
            (df['data'] <= pd.to_datetime(data_final))
        )
        df = df.loc[mask].reset_index(drop=True)

        if csv_saida:
            df.to_csv(csv_saida, index=False, encoding='utf-8-sig')
        logger.info(f"   [API] IPCA mensal IBGE (tab.1737): {len(df)} obs")
        return df
    except Exception as e:
        logger.warning(f"   [AVISO] IPCA mensal IBGE indisponivel: {e}")
        return pd.DataFrame(columns=['data', 'ipca_mensal'])


def carregar_olinda_credito(csv_saida, top=50000):
    """
    Carrega taxas de juros e inadimplencia por modalidade/instituicao
    via BACEN Olinda API - TaxasJurosMensalPorInstituicaoFinanceira.
    Tenta primeiro com top completo; em caso de falha, tenta com top reduzido.
    """
    if csv_saida and Path(csv_saida).exists():
        df = pd.read_csv(csv_saida)
        logger.info(f"   [CSV] Olinda credito: {len(df):,} obs (cache local)")
        return df

    url = (
        "https://olinda.bcb.gov.br/olinda/servico/taxas-juros/versao/v2/odata/"
        "TaxasJurosMensalPorInstituicaoFinanceira"
    )
    select = (
        "Mes,InstituicaoFinanceira,Segmento,Modalidade,"
        "TaxaJurosAoMes,TaxaJurosAoAno,TaxaInadimplencia,"
        "NumeroDeContratos,BaseDeCalculo"
    )
    tentativas = [top, 10000] if top > 10000 else [top]  # tenta grande, recua se necessario
    ultimo_erro = None

    for tentativa_top in tentativas:
        params = {
            "$top": tentativa_top,
            "$format": "json",
            "$select": select,
            "$orderby": "Mes desc",
        }
        try:
            logger.info(f"   [API] Olinda: solicitando {tentativa_top:,} registros...")
            resp = requests.get(url, params=params, timeout=120,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            registros = resp.json().get('value', [])
            if not registros:
                raise ValueError("API Olinda retornou lista vazia.")
            df = pd.DataFrame(registros)
            if csv_saida:
                df.to_csv(csv_saida, index=False, encoding='utf-8-sig')
            logger.info(f"   [API] Olinda credito: {len(df):,} obs")
            return df
        except Exception as e:
            ultimo_erro = e
            logger.warning(f"   [AVISO] Olinda (top={tentativa_top:,}): {e}")

    logger.error(f"   [ERRO] Olinda indisponivel apos todas as tentativas: {ultimo_erro}")
    return pd.DataFrame()


def carregar_pib_ibge(data_inicial=2015, data_final=2025, csv_saida=None):
    """
    Busca PIB per capita (R$ correntes) via IBGE SIDRA - anual.
    Tabela 6783 (PIB per capita), variavel 9812.
    """
    if csv_saida and Path(csv_saida).exists():
        df = pd.read_csv(csv_saida)
        df['data'] = pd.to_datetime(df['data'])
        logger.info(f"   [CSV] PIB per capita IBGE: {len(df)} obs (cache)")
        return df

    url = (
        "https://servicodados.ibge.gov.br/api/v3/agregados/6783"
        "/periodos/all/variaveis/9812?localidades=BR"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        series = _extrair_serie_ibge(resp.json())

        df = pd.DataFrame(list(series.items()), columns=['ano', 'pib_per_capita'])
        df['data'] = pd.to_datetime(df['ano'] + '-01-01')
        df['pib_per_capita'] = pd.to_numeric(df['pib_per_capita'], errors='coerce')
        df = (df[['data', 'pib_per_capita']]
              .sort_values('data')
              .reset_index(drop=True))
        df = df[
            (df['data'].dt.year >= data_inicial) &
            (df['data'].dt.year <= data_final)
        ].reset_index(drop=True)

        if csv_saida:
            df.to_csv(csv_saida, index=False, encoding='utf-8-sig')
        logger.info(f"   [API] PIB per capita IBGE: {len(df)} obs anuais")
        return df
    except Exception as e:
        logger.warning(f"   [AVISO] PIB IBGE indisponivel: {e}")
        return pd.DataFrame(columns=['data', 'pib_per_capita'])

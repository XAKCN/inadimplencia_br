# Análise Exploratória de Dados — Mapa da Inadimplência Brasileira

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

EDA completa sobre a **inadimplência bancária no Brasil**, construída com dados reais extraídos diretamente das APIs públicas do **Banco Central do Brasil (BACEN)** e do **IBGE**. O projeto cobre dois ângulos complementares: o comportamento macroeconômico agregado e a granularidade por modalidade de crédito e instituição financeira.

---

## Sobre o Projeto

O pipeline coleta, limpa e analisa séries temporais mensais de 2015 a 2025, cruzando inadimplência com Selic, IPCA, desemprego, PIB per capita, concessões e saldo de crédito. A análise por modalidade (dados Olinda) desce ao nível de produto bancário: cartão rotativo, cheque especial, crédito consignado, financiamento de veículos, entre outros.

Toda a saída é consolidada em dois painéis PNG de alta resolução, além das imagens individuais de cada gráfico — todos salvos diretamente em `outputs/`.

---

## Fontes de Dados

| Fonte | Endpoint | O que coleta |
|---|---|---|
| **BACEN SGS** | `api.bcb.gov.br` | Inadimplência Total, PF e PJ; Selic a.a.; IPCA acumulado 12m; Concessões de crédito (R$ bi); Saldo de crédito (R$ bi) |
| **BACEN Olinda** | `olinda.bcb.gov.br` | Taxas de juros ao mês/ano, inadimplência, número de contratos e base de cálculo — granularidade por modalidade × segmento × instituição |
| **IBGE SIDRA** | `servicodados.ibge.gov.br` | Taxa de desocupação PNAD Contínua (trimestral → interpolado mensal); PIB per capita anual (tab. 6783); IPCA variação mensal (tab. 1737) |

Os dados são cacheados em `/data` após a primeira requisição. Nas execuções seguintes, o pipeline lê do CSV local sem bater nas APIs.

---

## Estrutura do Repositório

```text
inadimplencia_br/
├── eda_br_inadimplencia.py     # Orquestrador principal — roda os dois painéis
├── requirements.txt
├── src/
│   ├── config.py               # Códigos SGS, período, limiares e parâmetros de saída
│   ├── constants.py            # MESES_PT, paleta de cores
│   ├── pipeline_utils.py       # configurar_exibicao(), montar_painel_sgs()
│   ├── data_loaders.py         # Requisições BACEN SGS, BACEN Olinda, IBGE SIDRA
│   ├── exploratory_utils.py    # Limpeza, outliers, estatística descritiva, dados faltantes
│   ├── plots_analise.py        # Gráficos para o dataset Olinda (modalidades)
│   ├── plots_macro.py          # Gráficos para o painel macroeconômico (SGS + IBGE)
│   └── visualization_utils.py  # PainelSaida — consolida gráficos individuais em painel PNG
├── data/                       # Cache CSV (gerado na 1ª execução)
└── outputs/                    # Painéis consolidados e imagens individuais
```

---

## Como Executar

### 1. Ambiente

```bash
git clone https://github.com/SeuUsuario/inadimplencia_br.git
cd inadimplencia_br

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Rodar a EDA completa

```bash
python eda_br_inadimplencia.py
```

Gera dois painéis em `outputs/`:

| Arquivo | Conteúdo |
|---|---|
| `painel_eda_macro.png` | Painel macroeconômico — BACEN SGS + IBGE |
| `painel_eda_inadimplencia.png` | Painel por modalidade/segmento — BACEN Olinda |

As imagens individuais de cada gráfico também ficam em `outputs/`.

---

## Análises Geradas

### Painel Macroeconômico (`plots_macro.py`) — 22 gráficos

**EDA base:**
- **Contexto econômico:** séries de desemprego PNAD, PIB per capita e IPCA mensal com médias móveis e área sombreada
- **Boxplots e distribuições:** dispersão e assimetria de cada indicador (Selic, IPCA, inadimplência PF/PJ/total, concessões, saldo de crédito)
- **Distribuições temporais:** inadimplência média por mês do ano e por ano, revelando sazonalidade
- **Séries temporais completas:** todas as variáveis com MM12 sobrepostas
- **Distribuição da inadimplência:** histograma + KDE com classificação por quartis (Baixa / Média-Baixa / Média-Alta / Alta)
- **Bivariadas:** scatter com regressão linear — inadimplência vs Selic, IPCA, desemprego, concessões e saldo
- **Inadimplência por período:** boxplot por ano, sazonalidade mensal, séries por ano sobrepostas, variação p.p. e comparação com MM12
- **Matriz de correlação:** heatmap triangular inferior incluindo lags (1, 3, 6, 12m) e médias móveis
- **Análises avançadas:** autocorrelação Lag1, MM3 vs MM12, heatmap Ano×Mês, violin por ano

**Gráficos de Análise Extensiva:**
- **PF vs PJ:** série comparativa, spread (área preenchida) e ratio PF/PJ ao longo do tempo
- **Variação YoY:** nível da série + barras de variação ano a ano em p.p. (verde = queda, vermelho = alta)
- **Carteira vs inadimplência:** saldo de crédito (barras, eixo esquerdo) sobreposto à taxa de inadimplência (linha, eixo direito)
- **Scatter matrix:** pairplot de todas as variáveis macro com KDE na diagonal
- **Regimes:** série colorida por regime de inadimplência (Baixa / Média / Alta) com limiares P33 e P66
- **Decomposição STL:** separação da série em tendência, sazonalidade e resíduo (requer `statsmodels`)
- **ACF / PACF:** autocorrelação e autocorrelação parcial até lag 24 (requer `statsmodels`)
- **OLS padronizado:** coeficientes beta com intervalo de confiança 95% e significância estatística (requer `statsmodels`)

### Painel por Modalidade (`plots_analise.py`) — 20 gráficos

**EDA base:**
- **Boxplots:** distribuição de TaxaJurosAoMes, TaxaJurosAoAno, TaxaInadimplencia, NumeroDeContratos e BaseDeCalculo
- **Distribuições numéricas:** histograma + KDE para cada variável numérica
- **Distribuições categóricas:** top 10 por volume — Segmento, Modalidade, Instituição
- **Variável-alvo:** histograma + KDE de TaxaInadimplencia com classificação por quartis
- **Bivariadas numéricas:** scatter com regressão para cada variável numérica
- **Bivariadas categóricas:** ranking horizontal por inadimplência média (Segmento, Modalidade)
- **Temporal:** evolução mensal da inadimplência com MM6
- **Matriz de correlação:** entre todas as variáveis numéricas Olinda
- **Análise avançada:** top 15 modalidades, boxplot por segmento, scatter juros×inadimplência, heatmap Ano×Segmento

**Gráficos de Análise Extensiva II:**
- **Ranking de instituições:** top 10 maior e top 10 menor inadimplência média lado a lado
- **Bubble chart:** Juros ao Mês × Inadimplência × Volume de contratos — uma bolha por modalidade
- **Evolução top 5 modalidades:** linhas múltiplas das 5 modalidades com maior inadimplência ao longo do tempo
- **Violin por segmento:** violin + boxplot de TaxaInadimplencia para cada segmento (PF, PJ, Gov)
- **Heatmap Instituição × Modalidade:** inadimplência média para as top 15 instituições × top 10 modalidades
- **Concentração de mercado (Pareto):** participação % por instituição com curva acumulada e linha 80%
- **Quartis por modalidade:** faixa IQR + mediana ao longo do tempo para as 6 modalidades de maior risco
- **Juros por segmento:** boxplot e média de TaxaJurosAoAno por segmento

---

## Configuração

Todos os parâmetros centralizados em `src/config.py`:

```python
# Período de coleta
PERIODO = {'data_inicial': '01/01/2015', 'data_final': '31/12/2025', ...}

# Limiares estatísticos
LIMIARES = {'outlier_iqr': 1.5, 'outlier_zscore': 3.0, 'corr_forte': 0.7, ...}

# Saída
SAIDA = {'dpi_miniatura': 110, 'dpi_painel': 150, 'olinda_top': 50_000}
```

---

## Dependências

```text
pandas >= 1.3
numpy >= 1.21
matplotlib >= 3.4
seaborn >= 0.11
scipy >= 1.7
requests >= 2.28
statsmodels >= 0.13   # STL, ACF/PACF, OLS (gráficos avançados de séries temporais)
```

---

## Licença

Distribuído sob a licença [MIT](https://opensource.org/licenses/MIT).

Projeto de estudo em Data Science Financeira e Quant. Sinta-se à vontade para abrir issues, pull requests ou adaptar para outros países e bases de dados públicas.

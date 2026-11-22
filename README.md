# Monitoramento e previsão do consumo energético em centros de dados

[![CI](https://github.com/RonyRangelSS/data-energy-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/RonyRangelSS/data-energy-monitor/actions/workflows/ci.yml)

Trabalho de Conclusão de Curso de Engenharia de Software (UPE Garanhuns) — "Sistema de monitoramento e previsão do consumo energético em centros de dados utilizando técnicas de machine learning".

O sistema:

1. prevê o consumo de energia de cada servidor e do total do data center para 1 h e 24 h à frente;
2. estima o consumo esperado para a carga atual de cada servidor e gera alertas quando há consumo não explicado pela carga de forma persistente;
3. exibe tudo em painéis em tempo quase real sobre reprodução acelerada de um dataset histórico real;
4. registra experimentos reproduzíveis que comparam os modelos com baselines.

> O sistema não age sobre os servidores e não economiza energia por si só. Ele mede, prevê e aponta consumo não explicado pela carga e o custo associado.

## Visão geral

```
dataset (CSV) ──► preparação ──► experimentos E1–E6 ──► relatórios JSON e modelos treinados
                                                              │
reprodutor ──HTTP──► API (FastAPI) ──► PostgreSQL ──► painel (Streamlit)
```

- **Fluxo offline:** prepara os dados, treina e avalia os modelos e grava os resultados em `artifacts/`.
- **Fluxo online:** o reprodutor envia o dataset à API em velocidade acelerada. A API calcula o consumo esperado, o desvio, os alertas e as previsões, e o painel consulta a API a cada 5 s.

Os dois fluxos usam o mesmo código de preparação. O sistema online depende dos modelos treinados no fluxo offline.

## Só quer ver os resultados?

Os relatórios dos experimentos (`artifacts/experiments/*.json`) e os gráficos (`artifacts/figures/*.png`) estão versionados. Para consultá-los, não é preciso rodar nada. Cada relatório guarda a configuração usada, o commit, as versões dos pacotes e o hash SHA-256 do arquivo de dados.

## Requisitos

- Windows, Linux ou macOS (os comandos abaixo são para o PowerShell)
- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/), que instala o Python 3.12 e as dependências fixadas em `uv.lock`
- [Docker Desktop](https://www.docker.com/products/docker-desktop/), para o PostgreSQL e o sistema completo
- 8 GB de RAM livres para os experimentos (o D1 tem 2,7 milhões de linhas) e cerca de 2 GB de disco para os dados

## Passo a passo

### 1. Clonar e instalar

```powershell
git clone https://github.com/RonyRangelSS/data-energy-monitor.git
cd data-energy-monitor
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # só se o uv não estiver instalado
uv sync
```

### 2. Baixar os dados

Dataset de Ilager et al. (2023), publicado no Zenodo (DOI [10.5281/zenodo.10069402](https://doi.org/10.5281/zenodo.10069402)) sob licença CC BY 4.0. Os arquivos **não** fazem parte do repositório.

```powershell
Invoke-WebRequest "https://zenodo.org/records/10069402/files/dataset1_combined.csv?download=1" -OutFile data/raw/dataset1_combined.csv
Invoke-WebRequest "https://zenodo.org/records/10069402/files/dataset2_combined.csv?download=1" -OutFile data/raw/dataset2_combined.csv
```

| Arquivo | Conteúdo | Tamanho |
|---|---|---|
| `dataset1_combined.csv` | D1, 81 servidores Dell R840 (conjunto principal) | 613 MB |
| `dataset2_combined.csv` | D2, 64 servidores Dell C6320 (replicação) | 184 MB |

### 3. Configurar o ambiente

```powershell
Copy-Item .env.example .env
```

Abra o `.env` e troque `POSTGRES_PASSWORD` e `API_KEY`. A senha aparece também em `DATABASE_URL` e em `TEST_DATABASE_URL`, então troque nas três linhas. O `.env` nunca é versionado.

| Variável | Uso |
|---|---|
| `DATABASE_URL` | Banco de desenvolvimento (porta 5434) |
| `TEST_DATABASE_URL` | Banco de teste, separado e apagado a cada reinício (porta 5435) |
| `API_KEY` | Chave exigida em `POST /readings` e `DELETE /scenarios/{nome}` (cabeçalho `X-API-Key`) |
| `TARIFF_KWH` | Tarifa usada no custo associado (por kWh) |
| `DISPLAY_TIMEZONE` | Fuso de exibição; os dados são gravados em UTC |

### 4. Subir o banco

```powershell
docker compose up -d db db_test
uv run alembic upgrade head
```

### 5. Rodar os testes

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov
```

Os testes de integração usam o banco `db_test`, que precisa estar no ar. Para rodar só os que não usam o banco:

```powershell
uv run pytest -m "not integration"
```

Os testes dos gráficos dependem dos relatórios em `artifacts/experiments/` e são ignorados se eles não existirem.

### 6. Reproduzir os experimentos

Rode as etapas **nesta ordem**. Cada uma depende das anteriores.

| # | Etapa | Comando | Saída principal |
|---|---|---|---|
| 1 | E1: qualidade dos dados e divisão temporal | `uv run python -m energy_monitor.experiments.e1_data_quality` | `e1_data_quality_d1.json`, `e1_data_quality_d2.json` e as grades em `data/processed/` |
| 2 | E5: seleção na validação | `uv run python -m energy_monitor.experiments.e5_forecasting --stage selection` | `e5_forecasting_selection_*.json` |
| 3 | E5: congelamento do protocolo | `uv run python -m energy_monitor.experiments.protocol_freeze --config configs/e5_forecasting.yaml` | `protocol_freeze.json` |
| 4 | E5: avaliação no teste | `uv run python -m energy_monitor.experiments.e5_forecasting --stage final` | `e5_forecasting_test_*.json` e os modelos de previsão |
| 5 | E2: seleção do consumo esperado | `uv run python -m energy_monitor.experiments.e2_expected_consumption --stage selection` | `e2_expected_consumption_selection_d1.json` |
| 6 | E3: seleção do detector | `uv run python -m energy_monitor.experiments.e3_detection --stage selection` | `e3_detection_selection_d1.json` |
| 7 | E2/E3: congelamento do protocolo | `uv run python -m energy_monitor.experiments.protocol_freeze --config configs/e2_e3_detection.yaml` | `protocol_freeze_e2_e3.json` |
| 8 | E2: avaliação no teste | `uv run python -m energy_monitor.experiments.e2_expected_consumption --stage final` | `e2_expected_consumption_test_d1.json` e o modelo de consumo esperado |
| 9 | E3: avaliação no teste | `uv run python -m energy_monitor.experiments.e3_detection --stage final` | `e3_detection_test_d1.json` |
| 10 | E4: energia, ociosidade e custo | `uv run python -m energy_monitor.experiments.e4_indicators` | `e4_indicators_d1.json` |
| 11 | E6: desempenho do sistema | com a API no ar (passo 7): `uv run python -m energy_monitor.experiments.e6_system` e `uv run python -m energy_monitor.experiments.e6_system --queries` | `e6_system_d1.json`, `e6_system_queries_d1.json` |
| 12 | Gráficos | `uv run python -m energy_monitor.experiments.figures` | PNGs em `artifacts/figures/` |

Regras do protocolo:

- Hiperparâmetros, variantes e limiares são escolhidos **só na validação** (etapas 2, 5 e 6).
- O congelamento grava o commit e o hash da seleção, e por isso exige o código commitado, sem alterações pendentes.
- A avaliação no teste falha se o protocolo não estiver congelado ou se a seleção mudar depois do congelamento. Assim, o teste é usado uma única vez.
- Os modelos treinados ficam em `artifacts/models/`, fora do git.
- Os gráficos saem dos relatórios JSON. Os da quebra estrutural e da escala do z também usam a grade do D1 e as estatísticas dos resíduos, e só são gerados quando esses arquivos existem.

### 7. Subir o sistema completo

É preciso ter concluído as etapas 1 a 9, que produzem os modelos usados em operação.

```powershell
docker compose --profile app up -d --build
```

- API: http://localhost:8000 (documentação interativa em http://localhost:8000/docs)
- Painel: http://localhost:8501

Para reproduzir o dataset sobre o sistema (cenário `demo`, 3 dias de dados a 600×, ou seja, 10 min de dados por segundo):

```powershell
docker compose --profile app --profile replay run --rm replayer
```

O período, a velocidade e o cenário podem ser trocados no `.env` com `REPLAY_START`, `REPLAY_END`, `REPLAY_SPEED` e `REPLAY_SCENARIO`. Cada execução grava o nome do cenário em leituras, previsões e alertas. Para apagar um cenário:

```powershell
Invoke-RestMethod -Method Delete -Uri http://localhost:8000/scenarios/demo -Headers @{ "X-API-Key" = "sua-chave" }
```

### Alternativa: sistema fora do Docker

Com o banco do passo 4 no ar, use um terminal para cada processo:

```powershell
uv run uvicorn --factory energy_monitor.api.app:create_app --port 8000
uv run streamlit run src/energy_monitor/dashboard/app.py
uv run python -m energy_monitor.replayer.cli --scenario demo --start 2019-05-30T12:20:00+00:00 --end 2019-06-02T12:20:00+00:00 --speed 600
```

O reprodutor também injeta anomalias a partir de um plano YAML com semente (`--injection plano.yaml`). As anomalias injetadas são registradas em `artifacts/experiments/replay_events_<cenário>.json` (ou no caminho de `--event-log`), e a API não vê essa informação.

## API

| Rota | Descrição |
|---|---|
| `POST /readings` | Recebe lotes de leituras (chave em `X-API-Key`). Valida, descarta duplicadas, calcula consumo esperado, desvio e alertas, e gera previsões a cada novo instante. |
| `GET /servers` | Servidores do cenário, com a última leitura |
| `GET /servers/{id}/series` | Real, esperado, limite superior, z e previstos de um servidor |
| `GET /alerts` | Alertas com início, fim, z máximo, consumo não explicado e justificativa |
| `GET /indicators` | Energia, energia em ociosidade e custo associado. A tarifa (`tariff`) e o limiar de ociosidade são opcionais. |
| `GET /forecasts` | Previsões do total (ou de um servidor), com quantos servidores ficaram de fora |
| `GET /models` | Modelos em operação |
| `DELETE /scenarios/{nome}` | Apaga um cenário (chave em `X-API-Key`) |
| `GET /health` | Saúde da API e do banco |

## Estrutura

```
src/energy_monitor/
├── common/        configuração, esquema, preparação, variáveis, indicadores
├── models/        interface e registro; previsão, consumo esperado e detectores
├── experiments/   E1–E6, avaliação, congelamento do protocolo e gráficos
├── database/      modelos SQLAlchemy e sessão
├── api/           FastAPI: ingestão, alertas, previsão online, consultas
├── replayer/      leitura do dataset, envio acelerado e injeção de anomalias
└── dashboard/     painel Streamlit
alembic/           migrações do banco
configs/           configuração YAML dos experimentos
artifacts/         relatórios e gráficos (versionados) e modelos (fora do git)
data/              raw/ e processed/ (fora do git)
notebooks/         exploração inicial (sem saídas)
tests/             unit/, integration/ e system/
```

## Dados e citação

Ilager, S. et al. (2023). *A Data-driven Analysis of a Cloud Data Center: Statistical Characterization of Workload, Energy and Temperature*. UCC '23. DOI [10.1145/3603166.3632137](https://doi.org/10.1145/3603166.3632137).

## Licença

Código sob licença MIT (ver [LICENSE](LICENSE)). Dataset sob CC BY 4.0, de Ilager et al. (2023).

# Monitoramento e previsão do consumo energético em centros de dados

Trabalho de Conclusão de Curso de Engenharia de Software (UPE Garanhuns) — "Sistema de monitoramento e previsão do consumo energético em centros de dados utilizando técnicas de machine learning".

O sistema:

1. prevê o consumo de energia de cada servidor e do total do data center para 1 h e 24 h à frente;
2. estima o consumo esperado para a carga atual de cada servidor e gera alertas quando há consumo não explicado pela carga de forma persistente;
3. exibe tudo em painéis em tempo quase real sobre reprodução acelerada de um dataset histórico real;
4. registra experimentos reproduzíveis que comparam os modelos com baselines.

> O sistema não age sobre os servidores e não economiza energia por si só. Ele mede, prevê e aponta consumo não explicado pela carga e o custo associado.

## Dados

Ilager, S. et al. (2023). *A Data-driven Analysis of a Cloud Data Center: Statistical Characterization of Workload, Energy and Temperature*. UCC '23. DOI [10.1145/3603166.3632137](https://doi.org/10.1145/3603166.3632137).

Os arquivos estão no Zenodo (DOI [10.5281/zenodo.10069402](https://doi.org/10.5281/zenodo.10069402)), sob licença CC BY 4.0, e **não** fazem parte deste repositório. Baixe `dataset1_combined.csv` (D1, principal) e `dataset2_combined.csv` (D2, replicação) e coloque os dois em `data/raw/`.

## Requisitos

- [uv](https://docs.astral.sh/uv/), que instala o Python 3.12 e as dependências fixadas em `uv.lock`
- Docker Desktop
- Git
- Cerca de 8 GB de RAM para os experimentos

## Instalação (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
Copy-Item .env.example .env   # troque a senha e a chave de API
docker compose up -d db db_test
uv run alembic upgrade head
```

## Reprodução dos experimentos

Rode as etapas na ordem abaixo. Cada etapa grava um relatório JSON em `artifacts/experiments/` com a configuração, o commit, as versões dos pacotes e o hash dos dados. Os modelos treinados ficam em `artifacts/models/`, fora do git.

| Etapa | Comando | Saída |
|---|---|---|
| E1: qualidade e divisão | `uv run python -m energy_monitor.experiments.e1_data_quality` | `e1_data_quality_d1.json`, `e1_data_quality_d2.json` e as grades em `data/processed/` |
| E5: seleção (validação) | `uv run python -m energy_monitor.experiments.e5_forecasting --stage selection` | `e5_forecasting_selection_*.json` |
| E5: congelamento | `uv run python -m energy_monitor.experiments.protocol_freeze --config configs/e5_forecasting.yaml` | `protocol_freeze.json` |
| E5: teste | `uv run python -m energy_monitor.experiments.e5_forecasting --stage final` | `e5_forecasting_test_*.json` e os modelos de previsão |
| E2: seleção | `uv run python -m energy_monitor.experiments.e2_expected_consumption --stage selection` | `e2_expected_consumption_selection_d1.json` |
| E3: seleção | `uv run python -m energy_monitor.experiments.e3_detection --stage selection` | `e3_detection_selection_d1.json` |
| E2/E3: congelamento | `uv run python -m energy_monitor.experiments.protocol_freeze --config configs/e2_e3_detection.yaml` | `protocol_freeze_e2_e3.json` |
| E2/E3: teste | `uv run python -m energy_monitor.experiments.e2_expected_consumption --stage final` e depois `uv run python -m energy_monitor.experiments.e3_detection --stage final` | `e2_*_test_d1.json`, `e3_*_test_d1.json` |
| E4: indicadores | `uv run python -m energy_monitor.experiments.e4_indicators` | `e4_indicators_d1.json` |
| E6: sistema | com a API no ar: `uv run python -m energy_monitor.experiments.e6_system` | `e6_system_d1.json` |

O congelamento só funciona com o código commitado. A etapa de teste falha se o protocolo não estiver congelado ou se a seleção mudar depois dele.

## Sistema completo (Docker Compose)

Depois de rodar E1, E5, E2 e E3, os modelos operacionais estão em `artifacts/`. Então:

```powershell
docker compose --profile app up -d --build
```

- API: http://localhost:8000 (documentação em http://localhost:8000/docs)
- Painel: http://localhost:8501

Para reproduzir o dataset sobre o sistema (cenário `demo`, 600× por padrão):

```powershell
docker compose --profile replay run --rm replayer
```

Ou, fora do Docker, com a API local:

```powershell
uv run uvicorn --factory energy_monitor.api.app:create_app --port 8000
uv run python -m energy_monitor.replayer.cli --scenario demo --start 2019-05-30T12:20:00+00:00 --end 2019-06-02T12:20:00+00:00 --speed 600
uv run streamlit run src/energy_monitor/dashboard/app.py
```

## API

| Rota | Descrição |
|---|---|
| `POST /readings` | Recebe lotes de leituras (chave de API em `X-API-Key`). Valida, descarta duplicadas, calcula consumo esperado, desvio e alertas, e gera previsões a cada novo instante. |
| `GET /servers` | Servidores do cenário, com a última leitura |
| `GET /servers/{id}/series` | Real, esperado, limite superior, z e previstos de um servidor |
| `GET /alerts` | Alertas com início, fim, z máximo, consumo não explicado e justificativa |
| `GET /indicators` | Energia, energia em ociosidade e custo associado. A tarifa (`tariff`) e o limiar de ociosidade são opcionais. |
| `GET /forecasts` | Previsões do total (ou de um servidor), com quantos servidores ficaram de fora |
| `GET /models` | Modelos em operação |
| `DELETE /scenarios/{nome}` | Apaga um cenário (chave de API) |
| `GET /health` | Saúde da API e do banco |

## Testes e lint

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov
uv run pytest -m "not integration"   # só testes que não usam o banco
```

## Estrutura

```
src/energy_monitor/
├── common/        configuração, esquema, preparação, variáveis, indicadores
├── models/        interface e registro; previsão, consumo esperado e detectores
├── experiments/   E1–E6, avaliação, congelamento do protocolo
├── database/      modelos SQLAlchemy e sessão
├── api/           FastAPI: ingestão, alertas, previsão online, consultas
├── replayer/      leitura do dataset, envio acelerado e injeção de anomalias
└── dashboard/     painel Streamlit
alembic/           migrações do banco
configs/           configuração YAML dos experimentos
artifacts/         resultados de experimentos (versionados) e modelos (fora do git)
notebooks/         exploração inicial
tests/             unit/ e integration/
```

## Licença

Código sob licença MIT (ver [LICENSE](LICENSE)). Dataset sob CC BY 4.0, de Ilager et al. (2023).

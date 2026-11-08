# Monitoramento e previsão do consumo energético em centros de dados

Trabalho de Conclusão de Curso de Engenharia de Software (UPE Garanhuns) — "Sistema de monitoramento e previsão do consumo energético em centros de dados utilizando técnicas de machine learning".

O sistema:

1. prevê o consumo de energia de cada servidor e do total do data center para 1 h e 24 h à frente;
2. estima o consumo esperado para a carga atual de cada servidor e gera alertas quando há consumo não explicado pela carga de forma persistente;
3. exibe tudo em painéis em tempo quase real sobre reprodução acelerada de um dataset histórico real;
4. registra experimentos reproduzíveis que comparam os modelos com baselines.

> Status: Fases 0 (fundação) e 1 (dados) concluídas. Próxima: Fase 2 (previsão offline).

## Dados

Ilager, S. et al. (2023). *A Data-driven Analysis of a Cloud Data Center: Statistical Characterization of Workload, Energy and Temperature*. UCC '23. DOI [10.1145/3603166.3632137](https://doi.org/10.1145/3603166.3632137).

Arquivos no Zenodo: DOI [10.5281/zenodo.10069402](https://doi.org/10.5281/zenodo.10069402), licença CC BY 4.0. O dataset **não** está neste repositório. Baixe `dataset1_combined.csv` (D1, principal) e `dataset2_combined.csv` (D2, replicação) e coloque em `data/raw/`.

## Requisitos

- [uv](https://docs.astral.sh/uv/) (gerencia o Python 3.12 e as dependências fixadas em `uv.lock`)
- Docker Desktop (PostgreSQL de desenvolvimento e de teste)
- Git

## Instalação (PowerShell)

```powershell
# 1. Instalar o uv (uma vez)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Criar o ambiente com as versões fixadas
uv sync

# 3. Configurar variáveis de ambiente
Copy-Item .env.example .env
# edite .env e troque senha e chave

# 4. Subir os bancos de desenvolvimento (porta 5434) e de teste (porta 5435)
docker compose up -d

# 5. Aplicar as migrações no banco de desenvolvimento
uv run alembic upgrade head
```

## Preparação dos dados e experimento E1

```powershell
uv run python -m energy_monitor.experiments.e1_data_quality
```

O comando lê `configs/e1_data_quality.yaml` e, para o D1 e o D2, gera a grade de 10 min em `data/processed/` e o relatório de qualidade em `artifacts/experiments/`. O relatório registra a configuração, o commit, as versões dos pacotes e o hash do arquivo de dados.

## Testes e lint

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov
uv run pytest -m "not integration"   # só testes que não usam o banco
```

## Notebook exploratório

```powershell
uv sync --group notebook
```

Abra `notebooks/ilager_exploratory_analysis.ipynb` no VS Code e selecione o kernel `.venv`.

## Estrutura

```
src/energy_monitor/
├── common/        configuração, esquemas das leituras, preparação, variáveis
├── models/        interface, registro, previsão, consumo esperado, detectores
├── experiments/   E1–E6
├── database/     modelos SQLAlchemy, sessão, repositórios
├── api/           rotas e serviços (FastAPI)
├── replayer/      leitura do dataset, envio, injeção de anomalias
└── dashboard/    Streamlit
alembic/           migrações do banco
artifacts/         modelos treinados (fora do git) e resultados de experimentos
notebooks/         exploração inicial
tests/             unit/, integration/, system/
```

## Licença

Código sob licença MIT (ver [LICENSE](LICENSE)). Dataset sob CC BY 4.0, de Ilager et al. (2023).

# foco-llm

Investigação experimental da seleção de informações relevantes em modelos de linguagem, com ajuste de modelos pré-treinados e avaliação da generalização entre tarefas.

**Autor:** Mateus.

**Estado atual:** infraestrutura do piloto sintético implementada. Ainda não há baseline de LLM, treinamento ou conclusão científica. O artigo será escrito a partir dos experimentos concluídos.

## O que já funciona

- Geração reproduzível de problemas de aritmética, dedução e acompanhamento de objetos.
- Versões limpas e com três tipos de distratores, com resposta preservada.
- Validação independente do gabarito e dos fatos necessários.
- Separação de treino/validação/teste por problema, mantendo suas variantes juntas.
- Prompts sem gabarito, avaliação de previsões registradas e gráfico de dispersão pareada.
- Saídas identificadas pelo conteúdo, preservação de resultados anteriores e logs JSON.

## Instalação

Requisitos: Python 3.12 e Poetry 2.4.1. Não é necessário ter GPU para esta etapa.

```bash
poetry install
poetry run pre-commit install
```

Todas as dependências são gerenciadas pelo Poetry e fixadas em `poetry.lock`. Em terminais Windows, habilite UTF-8: `$env:PYTHONUTF8 = "1"`. O projeto usa caminhos relativos e leitura/gravação explícita em UTF-8.

## Gerar o piloto

```bash
poetry run python -m foco_llm prepare --problems-per-task 100 --seed 42
```

Esse comando cria **300 problemas-base e 1.200 exemplos**: três tarefas, cem problemas por tarefa e quatro condições. A divisão é determinística: a cada dez problemas de cada tarefa, oito vão para treino, um para validação e um para teste.

As saídas ficam em `data/processed/<data-UTC>/<hash>/`:

- `dataset.json`: fatos, perguntas, respostas, evidências e splits.
- `prompts-train.json`, `prompts-validation.json`, `prompts-test.json`: entradas sem gabarito.
- `manifest.json`: versão, configuração, contagens e hash SHA-256 do dataset.
- `dataset-profile.png`: composição dos dados, **não desempenho de modelo**.

`data/processed/latest.json` aponta para a última publicação validada. Repetir o comando com os mesmos argumentos no mesmo dia reutiliza os artefatos idênticos. Em outra data, cria uma nova partição com o mesmo conteúdo e hash. Não há limpeza automática de resultados. Arquivos parciais de uma interrupção não são anunciados como uma publicação concluída; repetir a preparação reaproveita os arquivos íntegros.

No PowerShell, para localizar e validar a saída:

```powershell
$latestRun = Get-Content data/processed/latest.json -Raw | ConvertFrom-Json
$datasetPath = Join-Path (Join-Path 'data/processed' $latestRun.path) 'dataset.json'
poetry run python -m foco_llm validate $datasetPath
```

## Avaliar respostas registradas

O comando `score` **não executa um modelo**. Ele avalia um arquivo JSON com proveniência e respostas previamente coletadas:

```json
{
  "model_id": "identificador-do-modelo",
  "model_revision": "revisao-exata",
  "prompt_version": "evidence-json-v1",
  "seed": 42,
  "decoding": {"do_sample": false, "max_new_tokens": 128},
  "predictions": []
}
```

Cada resposta em `predictions` deve conter `id`, `answer` e `evidence`; o `id` deve corresponder a um exemplo do split avaliado. `answer` é uma string e `evidence` é uma lista de identificadores como `F1`. Não são aceitos IDs repetidos ou de outro split. Respostas ausentes entram como erros, inclusive uma execução sem respostas.

```powershell
poetry run python -m foco_llm score $datasetPath data/predictions.json --split test
```

As saídas ficam em `data/evaluations/<data-UTC>/<hash>/`: `report.json` e `paired-accuracy.png`. O gráfico usa Matplotlib, escala comum iniciada em zero e uma linha de igualdade. Cada ponto representa uma tarefa/condição; pontos sobrepostos são possíveis. As funções de visualização devolvem figuras e a exportação é feita separadamente, a 180 dpi.

## Qualidade e testes

```bash
poetry run ruff format .
poetry run ruff check --fix .
poetry run poe gate
```

O portão inclui formatação, lint, mypy estrito, pytest e varredura de segredos com falha em caso de achados. O mesmo comando roda no pre-commit e no CI. Dependabot monitora dependências. Os testes não acessam rede nem dados externos; arquivos de teste são criados em diretórios temporários.

## Estrutura

```text
src/foco_llm/
  core/        # geração, verificadores, prompts, métricas e gráficos
  models/      # contratos validados por Pydantic
  services/    # publicação e carregamento de artefatos
  utils/       # logging
tests/         # fronteiras, erros, regressões e fluxo integrado
docs/          # protocolo e diário de experimentos
data/          # saídas locais; não versionadas
```

## Limites e próximos passos

O piloto usa inglês nos enunciados, dificuldade fixa e templates compartilhados entre splits. Serve para verificar a infraestrutura; não sustenta afirmações de generalização estrutural ou de relevância semântica em texto livre. Os validadores reconhecem a gramática controlada do gerador.

Ainda faltam seleção do modelo, execução de inferência, ajuste, checkpoints de treinamento, alertas de execuções longas, diversidade de templates, intervalos de confiança e repetição por sementes. Não há agendamento nem consumo de serviços pagos nesta etapa.

- [Plano e referências](docs/research-plan.md)
- [Diário e registro de treinamentos](docs/experiment-log.md)

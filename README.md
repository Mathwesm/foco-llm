# foco-llm

Investigação experimental da seleção de informações relevantes em modelos de linguagem, com ajuste de modelos pré-treinados e avaliação da generalização entre tarefas.

**Autor:** Mateus.

**Estado atual:** primeiro baseline local concluído em 120 exemplos de validação. As 120 respostas foram rejeitadas pelo contrato de JSON puro por usarem blocos Markdown; a pontuação estrita zero não mede isoladamente raciocínio. [Resultados e limitações](docs/baseline-2026-09-27.md). Ainda não houve treinamento neste projeto. O artigo será escrito a partir dos experimentos concluídos.

**Diagnóstico adicional:** a avaliação versionada de conteúdo encontrou 37/120 respostas corretas quando o acerto é medido independentemente dos identificadores de evidência. É uma reanálise das mesmas respostas, não uma melhora do modelo. [Protocolo, comparação e gráfico](docs/content-evaluation-2026-09-27.md).

## O que já funciona

- Geração reproduzível de problemas de aritmética, dedução e acompanhamento de objetos.
- Versões limpas e com três tipos de distratores, com resposta preservada.
- Validação independente do gabarito e dos fatos necessários.
- Separação de treino/validação/teste por problema, mantendo suas variantes juntas.
- Prompts sem gabarito, avaliação de previsões registradas e gráfico de dispersão pareada.
- Saídas identificadas pelo conteúdo, preservação de resultados anteriores e logs JSON.

## Instalação

Requisitos: Python 3.12 e Poetry 2.4.1. Geração de dados e avaliação de respostas não exigem GPU; a inferência descrita abaixo foi executada em GPU NVIDIA.

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

## Executar o modelo localmente

Instale a camada opcional de inferência com `poetry install --with inference`. A configuração validada usa Python 3.12, PyTorch com CUDA 12.8 e GPU NVIDIA. Para CPU, informe `--device cpu`; o tempo e a precisão numérica podem diferir da execução em FP16 na GPU.

```powershell
poetry run python -m foco_llm.inference $datasetPath --revision 7ae557604adf67be50417f59c2c2f167def9a775 --output data/inference/2026-09-27
```

O padrão executa **validação**, sem abrir o teste final. O executor preserva cada resposta bruta e retoma checkpoints compatíveis ao repetir o mesmo comando. Mantenha o mesmo `--output` ao retomar em outro dia. A revisão do modelo, dados, código e ambiente ficam no manifesto. Não execute duas instâncias no mesmo destino.

O primeiro uso baixa o modelo público. Mesmo com os pesos em cache, o tokenizer desta versão pode consultar metadados na rede; `HF_HUB_OFFLINE=1` não foi compatível com o carregamento observado. A inferência acontece localmente e os enunciados não são enviados a um serviço de geração.

O diretório da execução contém `manifest.json`, `responses/`, `predictions.json`, `report.json`, `summary.json` e `paired-accuracy.png`. JSON inválido ou evidências inválidas ficam preservados e entram como falhas. Duração e memória são medidas pelo executor, com as limitações descritas no [plano por etapas](docs/next-steps.md).

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

Ainda faltam ajuste, checkpoints de treinamento, alertas de execuções longas, diversidade de templates, intervalos de confiança e repetição por sementes. Não há agendamento nem consumo de serviços pagos nesta etapa.

- [Plano e referências](docs/research-plan.md)
- [Diário e registro de treinamentos](docs/experiment-log.md)
- [Próximas etapas e critérios de conclusão](docs/next-steps.md)

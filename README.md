# foco-llm

Investigação experimental da seleção de informações relevantes em modelos de linguagem, com ajuste de modelos pré-treinados e avaliação da generalização entre tarefas.

**Autor:** Mateus.

**Seleção + aritmética — 01/10:** após 320 passos de treino específico,
o Qwen 1,5B selecionou todos os fatos relevantes em **6/10** problemas
com distratores, ante **0/10** dos braços locais anteriores. Usando essa
seleção, o cálculo por outro adaptador chegou a **5/10**; uma regra lexical
simples fez **10/10** neste benchmark sintético, limite importante para
as conclusões do TCC. [Resultados, gráfico e respostas](docs/arithmetic-selection-training-results-2026-10-01.md).

**Diagnóstico aritmético — 01/10:** em dez novas bases de validação, o
Qwen 1,5B treinado com subtotais acertou 10/10 expressões de uma operação,
mas só 1/10 de quatro operações no contrato original. Ao decompor as
mesmas expressões de quatro operações em chamadas curtas, acertou **9/10**
resultados finais; os casos com distrator semelhante seguem em 0/10.
[Método, respostas brutas, gráfico e limites](docs/arithmetic-diagnostic-results-2026-10-01.md).

**Verificação em novos casos — 30/09:** o melhor ajuste local (`mixed-15b`)
acertou **55/120** em novas instâncias sintéticas, contra **8/120** respostas
legíveis corretas do modelo original. Aritmética permaneceu em **0/40**.
[Protocolo, gráficos, respostas e limites](docs/fresh-instance-results-2026-09-30.md).

**Currículo aritmético — 30/09:** dois novos treinos com subtotais derivados
somente dos dados de treino deram 1/40 e 2/40 acertos aritméticos na validação.
O braço misto caiu de 57/120 para 41/120 acertos estritos e perdeu validade
de formato. [Resultados, custos e limites da comparação](docs/arithmetic-curriculum-results-2026-09-30.md).

**Nova rodada exploratória — 30/09:** cinco ajustes dirigidos/balanceados
foram avaliados na validação v2.1. O Qwen 1,5B misto alcançou **57/120**,
o Qwen 0,5B misto **52/120** e o TinyLlama 1,1B misto **44/120**.
Aritmética continuou fraca (respectivamente 1/40, 1/40 e 4/40).
[Protocolo, resultados por tarefa e saídas brutas](docs/targeted-training-results-2026-09-30.md).

**Rodada local encerrada — 30/09:** os nove treinos e a avaliação final
congelada terminaram. No teste reservado de 120 casos, C0/C1/C2/C3
acertaram respectivamente **6/16/26/9** respostas; C2 foi o melhor, mas
aritmética permaneceu em 0/40 e a transferência para acompanhamento foi
pequena. [Relatório final, limitações e artefatos auditáveis](docs/final-local-results-2026-09-30.md).
O próximo ambiente experimental é Kaggle; ajustes de parâmetros são uma
rodada opcional posterior, separada deste teste.

**Controles locais de 30/09:** nove treinos (C1/C2/C3 × três sementes) concluídos. Médias de acertos na validação: **31,7/120** com contexto limpo e supervisão da resposta, **34,3/120** com ruído similar e supervisão da resposta, **19,0/120** com ruído similar e supervisão de resposta + evidências; modelo original **6/120**. Aritmética segue em 0/40. [Resultados, figura e limitações](docs/local-controls-2026-09-30.md).

**Piloto local de tarefa-fonte — 30/09:** 64 passos LoRA em 32 exemplos de dedução com distratores; validação passou de **6/120 para 19/120**, principalmente na tarefa treinada. Uma semente, sem controles C1/C2 ainda: não é prova de transferência robusta. [Protocolo, resultados e gráficos](docs/source-task-pilot-2026-09-30.md).

**Diagnóstico de formato concluído:** 60 novas gerações; JSON explícito e resposta simples não resolveram a baixa acurácia limpa. [Resultado](docs/format-diagnostic-2026-09-29.md) e [pendências locais antes da nuvem](docs/local-completion-checklist.md).

**Diagnóstico de ordem:** nos 30 casos limpos, inverter as frases mudou cinco resultados de dedução (três perdas e dois ganhos), sem resolver a baixa competência inicial. [Método, pares e gráfico](docs/order-diagnostic-2026-09-28.md).

**Como entender os experimentos:** [guia dos 120 casos, tarefas, exemplos e histórico](docs/experiment-guide.md). A validação usa 30 problemas-base em inglês, cada um em quatro condições; os testes automatizados do código são uma verificação separada.

**Treino seguido de validação:** ciclo funcional v2.1 concluído; seis passos LoRA mantiveram 6/120 acertos, com evidências exatas de 9/120 para 10/120. [Relatório antes/depois](docs/adapter-validation-2026-09-28.md). Ainda não é evidência de eficácia do treinamento.

**Atualização do benchmark v2.1:** baseline sem ajuste concluído: **6/120 acertos**, com competência limpa abaixo do critério em todas as tarefas. Isso impede atribuir as falhas especificamente aos distratores. [Resultados, gráficos e próxima decisão](docs/baseline-v21-2026-09-28.md). A versão controla ordem relativa dos fatos e quantidade de frases; os comprimentos em tokens continuam diferentes.

**Histórico do teste funcional:** seis atualizações LoRA no modelo de 1,5B, pausa/retomada e recarga verificadas, com pico alocado de 3,41 GiB na GPU local. A execução retomada reproduziu os pesos e o estado do otimizador da execução contínua. [Relatório e gráfico](docs/training-smoke-2026-09-28.md). O artigo será escrito a partir dos experimentos concluídos.

**Baselines no piloto original:** Qwen2.5-0.5B-Instruct acertou 38/120 respostas e Qwen2.5-1.5B-Instruct, 74/120, ambos em BF16. [Comparação e limitações](docs/model-comparison-2026-09-27.md). Esses resultados não são misturados com os dados revisados usados no teste de treinamento.

**Histórico preservado:** o primeiro baseline de 0,5B em FP16 teve 37/120 respostas corretas na [análise de conteúdo](docs/content-evaluation-2026-09-27.md). A tentativa de 1,5B em FP16 apresentou [falha numérica](docs/numerical-failure-2026-09-27.md); por isso ambos foram repetidos em BF16 com proteção contra scores inválidos. Todas as saídas dos baselines válidos usaram blocos Markdown; pontuação estrita de JSON puro zero não significa zero acertos de conteúdo.

## O que já funciona

- Geração reproduzível de problemas de aritmética, dedução e acompanhamento de objetos.
- Versões limpas e com três tipos de distratores, com resposta preservada.
- Validação independente do gabarito e dos fatos necessários.
- Separação de treino/validação/teste por problema, mantendo suas variantes juntas.
- Prompts sem gabarito, avaliação de previsões registradas e gráfico de dispersão pareada.
- Saídas identificadas pelo conteúdo, preservação de resultados anteriores e logs JSON.
- Versão candidata dos dados com fatos embaralhados, tempos explícitos e templates/comprimentos separados por partição.
- Teste LoRA com máscara de perda, checkpoints íntegros, retomada e comparação com execução contínua.

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

Instale a camada opcional de inferência com `poetry install --with inference`. A configuração validada usa Python 3.12, PyTorch com CUDA 12.8 e GPU NVIDIA. Informe explicitamente `--precision bfloat16` para reproduzir a comparação atual; o padrão histórico da CLI continua sendo FP16. Para CPU, informe `--device cpu`; o backend usa FP32 nesse dispositivo.

```powershell
poetry run python -m foco_llm.inference $datasetPath --revision 7ae557604adf67be50417f59c2c2f167def9a775 --precision bfloat16 --output data/inference/2026-09-27
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

O piloto original usa inglês nos enunciados, dificuldade fixa e templates compartilhados entre splits. A versão candidata v2 separa templates e comprimentos de cadeia, mas muda ambos ao mesmo tempo e ainda precisa de auditoria ampliada. Os dados não sustentam afirmações de generalização estrutural ampla ou de relevância semântica em texto livre. Os validadores reconhecem a gramática controlada do gerador.

O ajuste funcional e os checkpoints já foram verificados. Ainda faltam treinamentos de eficácia, avaliação dos adaptadores, alertas de execuções longas, auditoria ampliada dos dados revisados, intervalos de confiança e repetição por sementes. Não há agendamento nem consumo de serviços pagos nesta etapa.

- [Plano e referências](docs/research-plan.md)
- [Diário e registro de treinamentos](docs/experiment-log.md)
- [Próximas etapas e critérios de conclusão](docs/next-steps.md)
- [Protocolo da comparação entre modelos](docs/model-comparison-protocol.md)
- [Controles de treinamento, repetições e nuvem](docs/training-and-cloud-plan.md)
- [Dados revisados e comandos do teste LoRA](docs/benchmark-v2-and-smoke.md)

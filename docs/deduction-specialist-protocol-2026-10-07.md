# Especialização em dedução: protocolo de 7B

## Pergunta

Um adaptador QLoRA treinado **somente para selecionar evidências de dedução** melhora a
seleção exata nessa tarefa? A melhora transfere para rastreamento, que não aparece no
treino, ou permanece especializada? Compararemos o Qwen2.5-7B-Instruct congelado e o
mesmo modelo com o adaptador. O resultado anterior do adaptador aritmético no conjunto
de transferência de 4/10 é exploratório e usa outro conjunto; não será tratado como
comparação pareada com este novo teste.

## Conjunto congelado antes do treinamento

`poetry run python scripts/run_deduction_specialist.py prepare --output <diretório>`
gera dados v2.1 com semente `20261007` e 300 problemas-base por tarefa. Seleciona
80 exemplos distintos da partição `train`, tarefa `deduction`, condição `similar`,
semente de seleção 42. Grava **apenas esses 80** em `training-dataset.json`. A partição
`test` fornece 30 bases por tarefa, cada qual com duas condições, `unrelated` e
`similar`: 60 casos de dedução e 60 de rastreamento em `holdout-dataset.json`.
As bases de treino e teste são disjuntas; o programa falha se não forem. Os hashes
SHA-256, os IDs selecionados e o tamanho do teste ficam em `dataset-manifest.json`.

Os enunciados são sintéticos e seguem a **mesma família de moldes** nas duas partições.
Portanto, acerto nesse conjunto demonstra generalização para novas instâncias desses
moldes, não para textos livres ou raciocínio humano em geral. A escolha do treino foi
motivada pelo desempenho baixo já observado em outro teste; por isso o experimento é
**pós-hoc**, não confirmação prospectiva da hipótese original.

## Treinamento

`poetry run python scripts/run_deduction_specialist.py train --output <diretório>`
usa a revisão fixa `a09a35458c702b33eeacc393d103063234e8bc28` do
Qwen2.5-7B-Instruct, NF4 de 4 bits e LoRA nos módulos de consulta/valor, com posto 4,
taxa de aprendizado `1e-4`, 320 atualizações, lote unitário e 80 exemplos repetidos
em ordem determinística por quatro ciclos. A perda supervisiona só a resposta JSON
com IDs de evidências. A pergunta visível usa exatamente o prompt da avaliação de
transferência; o gabarito não entra na pergunta. Há checkpoints verificáveis a cada
80 passos, passíveis de retomada. O custo de GPU e a perda de treino são registrados;
perda menor, isoladamente, não prova melhora no teste.

## Avaliação e interpretação

No Kaggle, rodar `scripts/run_cross_task_transfer.py evaluate` duas vezes sobre o
**mesmo** `holdout-dataset.json`: uma com `--backend cloud` sem adaptador e outra com
`--backend cloud --adapter <step-0320> --training-dataset
<training-dataset.json>`. O comando verifica a integridade do checkpoint e o hash do
conjunto de treino. A métrica principal é seleção exata dos IDs em dedução, separada
pelas duas condições. Rastreamento é a métrica secundária de transferência. Também
registrar validade do JSON, falsos positivos/negativos e respostas brutas. A comparação
pareada por base permite distinguir melhora real de mudança apenas de formato.

O conjunto original de 120 casos de transferência, já inspecionado durante o projeto,
pode ser uma análise secundária. Não usá-lo como confirmação cega deste novo treino.
Não misturar resultados de sementes, modelos ou conjuntos distintos numa diferença
causal simples. Se houver orçamento, repetir o **mesmo protocolo** com outra semente
antes de afirmar estabilidade; não mudar hiperparâmetros após ver este teste e chamar
o resultado de validação independente.

Em 7/10, a conta Kaggle mostrou `02:27 / 30 hrs` de GPU usadas na semana, deixando
aproximadamente 27h33 disponíveis. Conferir novamente antes de iniciar, pois a cota
pode mudar. O notebook deve guardar manifestos, histórico, resumos e respostas brutas
como saída da versão salva do Kaggle.

## Resultado da versão salva em 8/10

O [notebook Kaggle, versão 1](https://www.kaggle.com/code/matheusdsousaribeiro/foco-llm-especialista-em-deducao-qwen-7b/output?scriptVersionId=356224021)
terminou com sucesso em 2.997,5 s (49 min 57 s) na GPU T4 x2. O código usado foi o
commit `2f11d8b8b8642a68f266e7e0d5ed409eab2795fa`. Os artefatos completos estão
em `deduction-7b/` na saída da versão: `comparison.json`, manifestos, datasets,
`scores.json` por braço, respostas brutas e checkpoint. Hashes do dataset:
treino `153ff15972b62fb7e8372a312134918038ff709d05fbb2a90badde18881abab8`;
teste `1ca12dae3b0c101175f927ecbf36c3efc98c01160e44ae4ab9256e0ec79e9d4f`.
O treino completou 320 passos em 1.561,5 s com 80 exemplos únicos. A perda de treino
medida no *probe* caiu de 0,8549 para 0,00005; isso indica ajuste aos exemplos vistos,
sem medir generalização.

| Tarefa e distrator | Casos | Base: seleção exata | Especialista: seleção exata | Base: seleção válida | Especialista: seleção válida |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dedução, não relacionado | 30 | 0 | 4 | 21 | 30 |
| Dedução, cadeia semelhante | 30 | 1 | 2 | 26 | 30 |
| Rastreamento, não relacionado | 30 | 1 | 1 | 30 | 30 |
| Rastreamento, cadeia semelhante | 30 | 0 | 1 | 30 | 30 |

Na tarefa-alvo, a seleção exata passou de **1/60 para 6/60**. Na comparação dos
`scores.json` pareados, cinco entradas antes erradas ficaram exatas, nenhuma exata
regrediu e o acerto anterior permaneceu. Em rastreamento, a seleção exata passou de
**1/60 para 2/60**, com um ganho e uma perda entre as mesmas entradas. Nenhuma das
30 bases de dedução foi selecionada exatamente nas **duas** condições. Não se deve
comparar essas contagens diretamente aos 4/120 do seletor aritmético: os datasets,
históricos de treino e a hipótese testada são diferentes.

O especialista reduziu omissões de fatos relevantes em dedução de 220 para 110,
mas aumentou inclusões indevidas de 23 para 38. Em rastreamento, as omissões caíram
de 221 para 136 e as inclusões indevidas subiram de 39 para 82. A validade do formato
melhorou de 47/60 para 60/60 em dedução. Assim, parte do ganho decorre de respostas
válidas e seleção de mais fatos, não de filtragem precisa de distratores. Os 2/30
acertos de dedução com cadeia semelhante mostram que o principal caso difícil
continua sem solução satisfatória.

Esta é uma análise **pós-hoc, exploratória, de uma única semente e uma única família
de templates**. Treino e teste não repetem bases, mas compartilham moldes. As duas
condições de cada base são dependentes; 60 entradas de dedução não são 60 bases
independentes. Não há evidência suficiente aqui para afirmar transferência robusta,
entendimento de textos livres, significância estatística ou mecanismo interno de
raciocínio. Para avançar, a prioridade é uma avaliação de dedução com novos moldes e
redações antes de mais passos de treino sobre o mesmo conjunto.

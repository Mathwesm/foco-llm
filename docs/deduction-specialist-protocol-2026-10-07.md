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

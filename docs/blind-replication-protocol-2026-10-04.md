# Replicação cega de seleção aritmética — protocolo e resultados

O conjunto foi publicado em 4 de outubro de 2026, antes da inferência dos
modelos, em [`reports/2026-10-04/blind-replication/dataset.json`](../reports/2026-10-04/blind-replication/dataset.json).
SHA-256: `478f1a18b7eac35fcc8f727dc7629f10886ec03f075082d67902625c3f9275d5`.
Seed: `20261004`. São 60 problemas-base, cada um nas condições `explicit` e
`alias`, perfazendo 120 entradas pareadas. Os dois membros de cada par têm a
mesma conta e resposta; mudam as pistas de referência. O gerador usa o mesmo
formato sintético do teste cego anterior, mas cria novas instâncias numéricas.
Portanto, esta é uma **replicação no mesmo domínio**, não uma prova de
generalização entre tarefas ou textos naturais.

O gabarito tem cinco fatos: quantidade inicial e quatro atualizações do
recipiente perguntado. Outros cinco fatos são distrações. A métrica primária
é a seleção exata dos cinco IDs, sem IDs extras, separada por condição. As
métricas secundárias são validade do JSON, falsos positivos, falsos negativos
e acerto de um executor determinístico que calcula somente a partir dos fatos
selecionados. O acerto do executor **não mede aritmética interna da LLM**.

O prompt `selection_prompt_v2`, a geração gulosa, os modelos e os adaptadores
permanecem os mesmos do teste anterior. Compararemos cada modelo com sua
versão sem ajuste: Qwen2.5-1.5B-Instruct local e Qwen2.5-7B-Instruct no Kaggle.
Nenhum prompt, peso ou hiperparâmetro foi escolhido olhando este conjunto.
As 60 bases, e não as 120 entradas, são as unidades independentes para
interpretação de incerteza. Qualquer falha de infraestrutura deve ser
registrada, sem modificar o conjunto congelado.

Além da regra lexical antiga, há um controle simbólico mais forte: ele lê o
fato inicial visível, resolve o apelido declarado e seleciona as atualizações
desse mesmo recipiente. Esse controle acertou 60/60 na forma explícita e
60/60 na forma com apelido, sem ler os rótulos privados. O teto perfeito é uma
limitação importante: este template é solucionável por um parser especializado.
O resultado de uma LLM aqui mostra desempenho neste formato, não compreensão
geral. Respostas brutas, manifestações de execução e pontuações ficam em
`data/blind-replication/2026-10-04/`, fora do repositório.

## Resultados da replicação

Cada célula mostra acertos de seleção exata entre 60 casos da condição. O
executor determinístico acertou os mesmos totais, exceto nos dois resultados
locais assinalados abaixo. A comparação relevante para treino é cada modelo
contra sua própria versão base, mantendo os mesmos 60 pares e o mesmo prompt.

| Ambiente e modelo | Explícito | Apelido | Resposta do executor (explícito; apelido) |
| --- | ---: | ---: | ---: |
| Controle simbólico | 60/60 | 60/60 | 60/60; 60/60 |
| Local, Qwen2.5-1.5B-Instruct base | 3/60 | 0/60 | 3/60; 1/60 |
| Local, 1.5B com LoRA de 320 passos | 19/60 | 0/60 | 20/60; 1/60 |
| Kaggle, Qwen2.5-7B-Instruct base | 57/60 | 36/60 | 57/60; 36/60 |
| Kaggle, 7B com QLoRA de 320 passos | 60/60 | 46/60 | 60/60; 46/60 |

No local, houve 16 ganhos e nenhuma perda de seleção exata nos 60 casos
explícitos; nos casos com apelido, não houve ganho. No Kaggle, o 7B adaptado
ganhou 3 casos explícitos sem perder nenhum. Nos casos com apelido, ganhou 11,
mas perdeu 1 anteriormente correto: ganho líquido de 10. Os 120 resultados
brutos de cada checkpoint foram preservados, inclusive as respostas do modelo.
Todos os 240 retornos dos dois checkpoints do Kaggle foram JSON válido. O
baseline 7B teve 1 falso positivo e 2 falsos negativos na forma explícita;
na forma com apelido, 1 falso positivo e 28 falsos negativos. Após ajuste,
foram 0/0 e 0/17, respectivamente. Essas contagens de falsos positivos e
negativos são de **fatos**, não de perguntas.

O resultado em nuvem é a [versão concluída do notebook no Kaggle](https://www.kaggle.com/code/matheusdsousaribeiro/foco-llm-blind-replication-qwen-7b?scriptVersionId=355270958).
Os arquivos `scores.json`, `summary.json`, `manifest.json` e `responses/`
estão nos diretórios de saída `blind-replication-7b/baseline/3cee1d318ddcd79a/`
e `blind-replication-7b/adapted-0320/4d2c1213c5515834/` dessa versão.
Localmente, os mesmos artefatos estão em
`data/blind-replication/2026-10-04/local-baseline/c1dab8714a2e3175/`
e `data/blind-replication/2026-10-04/local-adapted-0320/cb4750e013772ffe/`.
O arquivo `reports/2026-10-04/blind-replication/manifest.json` registra o hash
do conjunto congelado. O notebook importado também confere esse hash antes de
avaliar.

A melhora do 7B ajustado na condição com apelido é um resultado replicado
**neste domínio sintético**. O 1.5B ajustado não resolveu essa condição. As
diferenças entre 1.5B e 7B não isolam o efeito do número de parâmetros:
checkpoint, quantização, ambiente e histórico de treinamento também diferem.
O executor explica por que um erro na seleção muda a resposta, mas não mede o
cálculo que a LLM faria sozinha. Também não se pode inferir, a partir deste
benchmark, generalização para textos naturais ou outras tarefas.

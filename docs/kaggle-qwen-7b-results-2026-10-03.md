# Qwen 2.5 7B no Kaggle: seleção de evidências aritméticas

Execução concluída em 2 de outubro de 2026, em aproximadamente 61 minutos.
[Versão 2 do notebook no Kaggle](https://www.kaggle.com/code/matheusdsousaribeiro/foco-llm-qwen-2-5-7b-qlora-640?scriptVersionId=354539788),
com os checkpoints, respostas brutas, gráfico e resumo completo nos outputs.
O [resumo numérico copiado da versão salva](../reports/2026-10-03/kaggle-qwen-7b/comparison-summary.json)
permanece no repositório para consulta independente do link temporário do Kaggle.
O [gráfico comparativo](../reports/2026-10-03/kaggle-qwen-7b/selection-comparison-7b.png)
foi reproduzido localmente a partir desse resumo.

## Desenho e interpretação da métrica

O modelo base é `Qwen/Qwen2.5-7B-Instruct`, revisão
`a09a35458c702b33eeacc393d103063234e8bc28`. O código executado veio
do commit `6b0478ca58b9df2baec91971aa7ba36af5d94545` da branch
`feat/task-balanced-training`. O Kaggle forneceu T4 x2, mas a implementação
usou somente a primeira GPU. O modelo foi quantizado em NF4 com dupla
quantização e computação FP32; o ajuste QLoRA treinou um seletor de
evidências por 640 passos, com checkpoints a cada 80 passos.

O treino usou 80 exemplos aritméticos da partição `train`, selecionados
deterministicamente. A avaliação usou 40 entradas da partição `validation`:
dez problemas-base, cada um apresentado em quatro formas (`text-1`,
`text-2`, `text-4` e `similar-4`). Portanto, **40 entradas não são 40
problemas-base independentes**. Uma conferência local do mesmo gerador e
configuração encontrou zero interseções de `base_id` e zero combinações
exatas de pergunta e fatos entre os 80 exemplos de treino e as 40
entradas de avaliação. Isso verifica a separação das entradas, mas não
elimina a semelhança dos templates sintéticos.

`selection_exact` exige que o modelo identifique todos e somente os IDs
dos fatos relevantes. `answer_correct` é a resposta de um **executor
determinístico** alimentado pelos fatos escolhidos. Não mede o modelo
fazendo a aritmética sozinho. O gabarito não foi incluído no prompt.

| Seleções exatas por dez entradas | Sem ajuste | Passo 320 | Passo 640 |
|---|---:|---:|---:|
| Texto, 1 operação | 10 | 10 | 10 |
| Texto, 2 operações | 9 | 10 | 10 |
| Texto, 4 operações | 7 | 10 | 10 |
| Distrator semelhante, 4 operações | 4 | 10 | 10 |
| **Total por 40 entradas** | **30** | **40** | **40** |

Neste conjunto, a resposta do executor teve as mesmas contagens de
acerto da seleção exata. A perda em exemplos de treino chegou perto de
zero; isso não é evidência de generalização. Os 320 passos adicionais
não melhoraram a métrica observada.

## Limites para o resumo expandido e o TCC

Estes resultados são **exploratórios**. As dez bases de validação já
haviam sido usadas em análises locais de prompts e treinamento; não são
um novo teste cego final. Os fatos dos exemplos seguem templates simples.
Um [controle lexical pós-hoc](arithmetic-selection-training-results-2026-10-01.md)
que seleciona frases pela repetição literal do recipiente-alvo já
alcançava 10/10 em todas as quatro formas. Assim, 40/40 mostra que o
modelo aprendeu a resolver **esta tarefa sintética de seleção**, mas não
supera o controle simples nem demonstra compreensão de textos naturais.

O experimento compara estados do mesmo 7B antes e depois do QLoRA. A
comparação com o 3B também altera a arquitetura e outras configurações;
não isola o efeito do número de parâmetros. Não houve avaliação do
adaptador 7B em uma segunda tarefa. Portanto o resultado **não comprova
generalização entre tarefas nem um mecanismo interno de raciocínio**.
Para o resumo expandido, descrevê-lo como estudo exploratório de seleção
de informações relevantes em problemas aritméticos. Para sustentar o
título amplo do TCC, ainda é necessária avaliação cega em outra tarefa,
com casos que não possam ser resolvidos por correspondência lexical.

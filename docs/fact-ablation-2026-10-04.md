# Ablação de fatos na seleção aritmética

Protocolo congelado em 4 de outubro de 2026, antes da inferência. O conjunto
[`dataset.json`](../reports/2026-10-04/fact-ablation/dataset.json) tem 20
problemas-base aritméticos, cada um apresentado em três versões: contexto
completo, remoção de uma atualização relevante e remoção de um fato
irrelevante. São 60 entradas, mas **20 unidades-base pareadas**, não 60
problemas independentes. O SHA-256 do conjunto é
`fa18508efd00462969a1ae5185a656e011b9bf1d664fb75be5c09dee6a17eef5`.

A pergunta e o prompt de seleção permanecem iguais entre as três versões de
cada base. Remover a atualização relevante muda o gabarito calculado a partir
do contexto restante e reduz em um o conjunto de IDs exigidos. Remover o
distrator preserva os IDs relevantes e a resposta. Esse desenho permite
observar se a seleção do modelo reage a uma mudança necessária e se permanece
estável diante de uma remoção desnecessária. Não permite observar diretamente
os estados internos ou o raciocínio do modelo.

O modelo retorna IDs de fatos em JSON. A seleção exata exige todos e somente
os IDs relevantes; a resposta aritmética é calculada **externamente** a partir
dos IDs escolhidos, sem corrigir a escolha. Logo, `answer_correct` mede o
pipeline seletor + executor, não cálculo mental da LLM. Geração gulosa,
modelo fixado por revisão e mesmo adaptador de 320 passos usado na replicação
cega; nenhum destes 60 itens ajustou os pesos. O controle simbólico foi
verificado na geração do conjunto: em todos os 20 pares, retirar o fato
relevante mudou a resposta correta e retirar o irrelevante a preservou.

## Resultado local: Qwen2.5-1.5B-Instruct

| Contexto | Base: seleção exata / resposta correta | Adaptado 320: seleção exata / resposta correta |
| --- | ---: | ---: |
| Completo | 2/20 · 2/20 | 9/20 · 9/20 |
| Sem atualização relevante | 0/20 · 1/20 | 6/20 · 6/20 |
| Sem fato irrelevante | 2/20 · 2/20 | 6/20 · 6/20 |

Todas as 120 respostas locais (60 por braço) eram JSON válido. A melhora de
2/20 para 9/20 no contexto completo indica ganho neste pequeno subconjunto;
as quedas nos outros contextos mostram sensibilidade, mas por si só não
distinguem atenção ao fato de fragilidade do prompt. Em particular, a remoção
do distrator **não** melhorou a taxa do adaptado: caiu de 9/20 para 6/20.
Portanto, não cabe afirmar que o modelo ignora consistentemente informação
irrelevante. Respostas brutas, pontuações e manifestos locais ficam em
`data/fact-ablation/2026-10-04/` (diretório ignorado pelo Git).

Na análise pareada do adaptador local, 9/20 bases eram corretas no contexto
completo. Após retirar o fato irrelevante, cinco dessas nove continuaram
corretas, quatro deixaram de ser e uma base antes errada passou a ser correta.
Esse saldo de 6/20 oculta mudanças individuais em ambas as direções.

## Replicação no Kaggle: Qwen2.5-7B-Instruct

O [notebook da ablação](https://www.kaggle.com/code/matheusdsousaribeiro/kaggle-fact-ablation-7b?scriptVersionId=355774583)
terminou com sucesso em 6 de outubro de 2026 (versão 4, 892,8 s de execução
do kernel), usando T4 x2 e o mesmo conjunto congelado. Os manifestos dos dois
braços registram Qwen2.5-7B-Instruct na revisão
`a09a35458c702b33eeacc393d103063234e8bc28` e SHA-256 da avaliação
`fa18508efd00462969a1ae5185a656e011b9bf1d664fb75be5c09dee6a17eef5`.
O braço adaptado usa o checkpoint de 320 passos com SHA-256 do adaptador
`e5912f975b7df0930dbe44bd5a6a898f87cafaed884358f3bd37aa261cba1023`.
O Kaggle publicou `summary.json`, `scores.json` e `manifest.json` em ambos.
As 120 respostas textuais brutas foram preservadas em
[`kaggle-7b-raw-responses.json`](../reports/2026-10-04/fact-ablation/kaggle-7b-raw-responses.json),
associadas aos IDs do conjunto congelado. Como o download direto do arquivo
falhou no navegador, as respostas foram transcritas da visualização dos dois
`scores.json` no Kaggle; cada sequência de 60 respostas foi comparada por
SHA-256 com a sequência salva localmente. Os hashes são
`7a90db443a92be83193d37482d1679a2d6d54734e8863f4514f33fa16a132328`
(base) e
`3a396a55e6b2c7fe1303d2b8e240e374797e834a50e95d5d883214b95cf4c811`
(adaptado). Os arquivos originais completos de pontuação e os checkpoints
individuais continuam no output do Kaggle.

| Contexto | Base 7B: seleção exata / resposta correta | Adaptado 7B: seleção exata / resposta correta |
| --- | ---: | ---: |
| Completo | 18/20 · 18/20 | 20/20 · 20/20 |
| Sem atualização relevante | 17/20 · 17/20 | 18/20 · 18/20 |
| Sem fato irrelevante | 17/20 · 17/20 | 19/20 · 19/20 |

As 120 respostas 7B eram JSON válido. A seleção do 7B adaptado caiu de
20/20 no contexto completo para 19/20 mesmo após retirar um distrator; isso
impede interpretar o teto inicial como invariância à informação irrelevante.
O 7B também perdeu dois acertos quando uma atualização relevante foi retirada,
embora o gabarito tenha mudado corretamente nos 20 casos. As taxas medem
seleção sob o prompt e o template usados, não pensamento interno. As duas
primeiras tentativas no Kaggle não produziram pontuações válidas (falha de
download de dependência e script ausente); a versão 4 corrigiu o empacotamento
do notebook sem alterar conjunto, prompt ou pesos.

## Limites de interpretação

Os textos são sintéticos e em inglês, com estrutura repetida. O prompt contém
um exemplo de formato que não varia nas intervenções. Não há múltiplas
sementes de treinamento, calibração de significância ou domínio médico.
Assim, estas contagens descrevem comportamento observável nos 20 pares e
servem como análise complementar da transferência entre tarefas, sem provar
um mecanismo de pensamento ou generalização ampla.

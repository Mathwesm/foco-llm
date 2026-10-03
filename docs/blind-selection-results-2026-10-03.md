# Teste cego de seleção aritmética — 3 de outubro de 2026

## Pergunta e protocolo congelado

Este teste examina se o seletor identifica fatos relevantes em problemas
aritméticos **novos**, sem escolher prompt, checkpoint ou hiperparâmetro após
ver as respostas. O conjunto foi gerado e congelado antes de qualquer
inferência dos modelos. Seu SHA-256 é
`1e53ae1df83a4282ea35ef2e59b387236761f9391ab0a4aa2dd825fbccc4c183`.
A [cópia do conjunto](../reports/2026-10-03/blind-selection/dataset.json)
contém 20 problemas-base e duas condições por base, totalizando 40 entradas.
As duas condições compartilham os mesmos valores, operações e resposta; mudam
as pistas textuais. Não há combinação exata de pergunta e fatos coincidente
com o treino ou a validação anterior. Os templates, contudo, continuam
sintéticos e aparentados aos exemplos de desenvolvimento.

Na condição `explicit`, cada uma das quatro operações repete literalmente o
nome do recipiente-alvo. Na condição `alias`, o fato inicial apresenta um
apelido para o alvo; as operações subsequentes usam somente esse apelido, e um
fato de outro recipiente contém a cor do alvo apenas na descrição da tampa.
O gabarito exige o fato inicial e as quatro operações do alvo. O prompt
`selection_prompt_v2` e o parser de JSON são os mesmos do piloto anterior.
Uma seleção é exata somente se contiver todos os cinco IDs relevantes e
nenhum irrelevante. Um executor determinístico calcula a resposta a partir
dos fatos escolhidos; seu acerto **não mede cálculo do modelo**. Arquivos de
respostas brutas e métricas são mantidos separadamente para cada braço em
`data/blind-selection/2026-10-03/`.

O controle lexical extrai a cor da pergunta e escolhe fatos que a repetem
literalmente. Ele foi especificado antes da inferência dos modelos. O conjunto
passou por validação estrutural automática: pares preservam a resposta, cada
gabarito contém cinco fatos, e o executor chega ao resultado correto quando
recebe a evidência de referência. As condições são medidas pareadas das mesmas
20 bases; não devem ser tratadas como 40 problemas independentes.

## Resultado local

Foi avaliado `Qwen/Qwen2.5-1.5B-Instruct` na revisão
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, sem ajuste e com o
adaptador de seleção após 320 passos de LoRA. Ambos receberam os mesmos 40
prompts, em BF16 e geração gulosa. O adaptador já havia sido treinado em 80
casos aritméticos; nenhum exemplo do conjunto cego foi usado no treino.

| Seletor | Seleção exata `explicit` | Seleção exata `alias` | Resposta do executor `explicit` | Resposta do executor `alias` |
| --- | ---: | ---: | ---: | ---: |
| Regra lexical | 20/20 | 0/20 | 20/20 | 0/20 |
| 1,5B sem ajuste | 0/20 | 0/20 | 0/20 | 0/20 |
| 1,5B com LoRA | 7/20 | 1/20 | 7/20 | 1/20 |

O ajuste melhorou o modelo em relação à sua própria versão base na forma
literal, mas a regra lexical ainda foi superior nessa condição. A forma com
apelido foi difícil para ambos os modelos locais: o adaptado acertou somente
uma das 20 bases. A correspondência das contagens de seleção e resposta
decorre do executor nesta amostra, e não de o modelo ter feito as contas.
Na forma `alias`, o adaptado omitiu 40 fatos relevantes e incluiu 12
irrelevantes ao longo das 20 entradas. Os resultados são uma avaliação
exploratória externa ao conjunto de desenvolvimento, mas permanecem limitados
por um único conjunto pequeno e por um gerador sintético.

No caso `blind-arithmetic-0000:alias`, por exemplo, o alvo azul é apresentado
como `primary jar`. O modelo adaptado selecionou os fatos `F1, F2, F5, F6,
F9, F10`; o gabarito era `F1, F3, F5, F7, F10`. Ele incluiu o valor inicial
do recipiente vermelho, cuja tampa era azul, e omitiu duas atualizações do
recipiente-alvo. O executor recusou a conta porque havia dois valores
iniciais. Esse exemplo mostra uma falha de seleção de contexto, antes de
qualquer avaliação da habilidade de somar ou subtrair.

## Resultado 7B no Kaggle

O [notebook do teste cego, versão 2](https://www.kaggle.com/code/matheusdsousaribeiro/foco-llm-blind-test-qwen-7b?scriptVersionId=354998927)
executou em GPU T4 x2, usando a primeira GPU. O mesmo modelo base
`Qwen/Qwen2.5-7B-Instruct`, revisão
`a09a35458c702b33eeacc393d103063234e8bc28`, foi avaliado sem ajuste e
com o adaptador de 320 passos do experimento anterior. A inferência usou
quantização NF4 com dupla quantização, cálculo em `float32`, geração gulosa
e os mesmos 40 prompts congelados; nenhum treinamento adicional ocorreu.
Na saída preservada da versão 2, `blind-selection-7b/baseline/ea01cf365b565f6f/`
e `blind-selection-7b/adapted-0320/5c9b43ebb43e8b50/` contêm, para cada
braço, `manifest.json`, `scores.json`, `summary.json` e as respostas brutas
em `responses/`.

| Seletor | Seleção exata `explicit` | Seleção exata `alias` | Resposta do executor `explicit` | Resposta do executor `alias` |
| --- | ---: | ---: | ---: | ---: |
| 7B sem ajuste | 18/20 | 13/20 | 18/20 | 13/20 |
| 7B com LoRA, 320 passos | 20/20 | 17/20 | 20/20 | 17/20 |

O adaptador acrescentou dois acertos na forma explícita e quatro na forma
com apelido em relação ao mesmo 7B sem ajuste. Na condição `alias`, ele não
incluiu fatos irrelevantes, mas omitiu quatro fatos relevantes distribuídos
pelos 20 casos; três seleções ficaram incompletas. A conta do executor
acompanha a seleção exata neste conjunto e não demonstra capacidade de
cálculo do modelo. São contagens descritivas em 20 bases pareadas, insuficientes
para uma conclusão geral sobre escala ou mecanismos internos.

## Interpretação e limites

O 7B adaptado superou a regra lexical na condição `alias` (17/20 contra
0/20). Isso mostra transferência para estas reformulações sintéticas, mas um
parser que resolve apelidos provavelmente também acertaria muitos casos. Tal
parser ainda é um controle a construir se o benchmark for ampliado. O teste
não mede compreensão de textos naturais, generalização entre tarefas ou um
mecanismo interno causal. As comparações entre 1,5B local e 7B na nuvem são
descritivas, pois diferem em tamanho, precisão, quantização e ambiente. O
mesmo vale para a comparação com a regra lexical, que foi deliberadamente
simples e falha por construção na condição de apelido.

# Teste cego de transferência entre tarefas

Protocolo congelado em 4 de outubro de 2026, antes da inferência. O conjunto
está em [`reports/2026-10-04/cross-task-transfer/dataset.json`](../reports/2026-10-04/cross-task-transfer/dataset.json)
e o hash no `manifest.json` ao lado. Há 30 problemas-base de dedução de
propriedades e 30 de rastreamento de objetos. Cada base aparece com um
distrator não relacionado (`unrelated`) e com uma segunda cadeia semelhante
(`similar`), totalizando 120 entradas pareadas. A semente é `20261004`;
somente o split de teste do gerador v2.1 foi selecionado. O treino do
adaptador de seleção usou **somente aritmética**, na semente 42. Nenhum
exemplo de dedução ou rastreamento deste conjunto é usado para ajuste.

O prompt pede um JSON com IDs dos seis fatos necessários, com o mesmo exemplo
de formato aritmético usado no treino. A instrução da tarefa muda para
definir a cadeia de propriedades ou de transferências. Essa mudança de
instrução é uma limitação: estamos medindo transferência sob um novo prompt,
não um mecanismo isolado. O modelo não recebe os IDs corretos nem a resposta.
Geração gulosa, revisão do modelo e adaptador de 320 passos são os do teste
cego aritmético. Comparamos o mesmo modelo com e sem adaptador, sem escolher
prompt ou pesos com base neste conjunto.

A métrica primária é a **seleção exata** dos seis IDs, sem extras, por tarefa
e condição. Validade do JSON, falsos positivos e falsos negativos por fato
são secundários. Um controle simbólico percorre as cadeias a partir de
frases visíveis e acertou 120/120, demonstrando que o conjunto é solucionável
por uma regra especializada. As 60 bases são as unidades independentes; as
duas condições de cada base são relacionadas. Preservamos saídas brutas,
pontuações e manifestos por checkpoint em `data/cross-task-transfer/2026-10-04/`
localmente e na saída da versão do Kaggle.

Este teste examina **seleção de evidências entre tarefas**, não o cálculo
interno, nem atividade neuronal, nem competência médica. Os textos são
sintéticos e em inglês. Resultados positivos sustentariam uma afirmação
comportamental limitada a dedução e rastreamento nesses templates;
resultados negativos delimitariam onde o ajuste aritmético não transfere.

## Resultado local

| Tarefa e distrator | 1,5B base: exata/válida | 1,5B LoRA 320: exata/válida |
| --- | ---: | ---: |
| Dedução, não relacionado | 0/30; 28/30 | 0/30; 27/30 |
| Dedução, cadeia semelhante | 0/30; 25/30 | 0/30; 25/30 |
| Rastreamento, não relacionado | 0/30; 15/30 | 0/30; 27/30 |
| Rastreamento, cadeia semelhante | 0/30; 2/30 | 0/30; 15/30 |

O ajuste melhorou a validade do formato em rastreamento, mas **não produziu
nenhuma seleção exata** nas duas tarefas. A contagem de fatos omitidos caiu de
153 para 124 e de 159 para 125 nas duas condições de dedução; em rastreamento,
de 165 para 99 e de 178 para 149. Essas medidas secundárias não convertem o
resultado primário em sucesso. Os resultados locais completos, com respostas
brutas, estão em `data/cross-task-transfer/2026-10-04/local-baseline/` e
`data/cross-task-transfer/2026-10-04/local-adapted-0320/`.

## Resultado no Kaggle: Qwen2.5-7B-Instruct

A [versão 2 do notebook](https://www.kaggle.com/code/matheusdsousaribeiro/kaggle-cross-task-transfer-7b/log?scriptVersionId=355278834)
terminou em 1.389,6 segundos com T4 x2 disponível. O mesmo checkpoint de
320 atualizações, treinado somente em seleção aritmética, foi comparado à
base 7B. A inferência utiliza uma GPU. Os arquivos `summary.json`,
`scores.json`, respostas e manifestos estão no diretório de saída
`cross-task-transfer-7b/` da versão salva.

| Tarefa e distrator | 7B base: exata/válida | 7B QLoRA 320: exata/válida | Fatos omitidos, base → adaptado | Fatos indevidos, base → adaptado |
| --- | ---: | ---: | ---: | ---: |
| Dedução, não relacionado | 0/30; 19/30 | 1/30; 26/30 | 121 → 91 | 1 → 3 |
| Dedução, cadeia semelhante | 0/30; 17/30 | 0/30; 27/30 | 134 → 99 | 8 → 32 |
| Rastreamento, não relacionado | 0/30; 30/30 | 3/30; 30/30 | 90 → 50 | 4 → 6 |
| Rastreamento, cadeia semelhante | 0/30; 30/30 | 0/30; 30/30 | 134 → 118 | 29 → 58 |

A seleção exata passou de **0/120 para 4/120**; as 120 entradas representam
60 problemas-base pareados. O ganho pequeno e a queda a zero na condição de
cadeia semelhante impedem afirmar transferência robusta. O adaptador melhorou
frequentemente a recuperação de fatos relevantes, mas incluiu mais fatos da
cadeia distratora em três das quatro condições. Isso explica por que métricas
parciais e validade de JSON não substituem a métrica primária.

Exemplos reais do conjunto congelado: no caso aritmético
`replication-arithmetic-0000:alias`, o recipiente violeta começa com 56
bolinhas e recebe -6, +6, +12 e +9, chegando a 77; o recipiente laranja com
tampa violeta é distrator. Em `v2-deduction-000009:similar`, a cadeia de cinco
relações parte de P5169 e chega a P6379, enquanto uma cadeia paralela âmbar
não responde à pergunta. Em `v2-tracking-000009:similar`, um objeto de cobre
parte da caixa B9102 e termina na B9442 após cinco transferências; a posição
de um objeto violeta é distratora. Não houve domínio médico nem avaliação de
compreensão clínica. Esses exemplos ilustram a tarefa, não constituem prova
de que os identificadores gerados sejam linguagem natural variada.

A intervenção controlada aqui é trocar um distrator não relacionado por uma
cadeia semelhante mantendo a base do problema. Ela mede sensibilidade a
contexto, mas não revela causalmente os estados internos ou o “pensamento” do
modelo. Remover individualmente fatos relevantes e irrelevantes é um teste de
ablação adicional; não o confundimos com esta comparação pareada.

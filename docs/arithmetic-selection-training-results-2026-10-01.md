# Treino local de seleção e cálculo a partir dos fatos escolhidos

O adaptador de seleção concluiu **320 atualizações LoRA** sobre **80 casos
aritméticos de treino com distratores**. A revisão base foi mantida;
somente os pesos LoRA mudaram, e a recarga reproduziu os logits do
checkpoint. O tempo de atualização medido foi 81,6 segundos, sem contar
carregamento, checkpoints e verificação. A perda do exemplo de treino caiu
de 0,4211 para 0,00013; isso apenas confirma que o objetivo foi aprendido
nos exemplos vistos. [Configuração e protocolo](arithmetic-selection-training-protocol-2026-10-01.md),
[manifest e resumo do treino](../reports/2026-10-01/arithmetic-selection/training-summary.json).

Na mesma validação exploratória de dez bases, o novo adaptador selecionou
exatamente os fatos necessários em **6/10** casos com distratores. Os
modelos original, misto e de subtotais fizeram **0/10** sob o mesmo prompt.
O executor determinístico, recebendo somente a seleção do novo adaptador,
acertou **6/10** respostas finais.

| Seleção exata por dez casos | Texto, 1 operação | Texto, 2 operações | Texto, 4 operações | Distrator, 4 operações |
|---|---:|---:|---:|---:|
| Qwen 1,5B original | 1 | 2 | 1 | 0 |
| Qwen 1,5B misto | 2 | 3 | 1 | 0 |
| Qwen 1,5B subtotais | 3 | 4 | 2 | 0 |
| **Qwen 1,5B seleção treinada** | **10** | **10** | **9** | **6** |
| Regra lexical pós-hoc | 10 | 10 | 10 | 10 |

O [gráfico](../reports/2026-10-01/arithmetic-selection/selection-accuracy.png)
mostra a comparação. Na condição com distrator, o seletor treinado
identificou 46 dos 50 fatos relevantes, omitiu 4 e incluiu 3 irrelevantes;
em um caso selecionou dois valores iniciais e o executor recusou a conta.
O arquivo de [pontuações e respostas brutas](../reports/2026-10-01/arithmetic-selection/evidence-only-15b-v2/scores.json)
permite inspecionar cada erro.

Para medir o **sistema com dois modelos**, passamos apenas as frases
escolhidas pelo seletor a um extrator sintático de números e operações;
o adaptador de subtotais executou cada operação em uma chamada. Nenhum ID
ou resposta de referência foi fornecido aos modelos. Se faltava uma das
quatro operações, o caso continuou no denominador como erro. O sistema
acertou **5/10** problemas com distratores: seis seleções completas
permitiram o cálculo; uma delas terminou em `91` quando o correto era
`90`, por `84 + 5 → 90` no terceiro passo. As [respostas por etapa](../reports/2026-10-01/arithmetic-selection/selected-chain-15b/scores.json)
e os 24 checkpoints brutos estão publicados.

O controle sintático sem modelo alcançou **10/10** em todas as formas:
bastou extrair da pergunta o nome literal do recipiente e selecionar as
frases que repetem esse nome. Isso revela uma limitação importante do
benchmark: ele é artificial e lexicalmente simples. O ganho de 0/10 para
6/10 mostra que o treino melhorou o *modelo* nessa validação, mas não
prova uma capacidade geral de compreender relevância em textos naturais;
nesse conjunto específico, a regra simples é superior e mais barata.

As dez bases por condição são uma amostra pequena e foram reutilizadas
em explorações de prompt e treino. Todos estes resultados são
**exploratórios**. O teste final congelado não foi aberto. Antes de
afirmar generalização, precisamos de novas bases e casos em que o alvo
não aparece literalmente em todos os fatos relevantes, além de comparar
com a regra lexical e repetir o treino com outras sementes.

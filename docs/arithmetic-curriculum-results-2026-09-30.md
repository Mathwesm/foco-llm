# Currículo de subtotais: resultado exploratório

O [protocolo](arithmetic-curriculum-protocol-2026-09-30.md) foi definido após a
matriz de cinco ajustes dirigidos. Os dois braços abaixo usaram apenas problemas
de **treino** e foram avaliados nos mesmos 120 casos de **validação** v2.1.
Cada tarefa contém 40 casos (dez problemas-base em quatro condições). O teste
final reservado da rodada anterior não foi reaberto.

| Braço | Exemplos de treino | Passos | Acertos estritos / 120 | Aritmética / 40 | Dedução / 40 | Acompanhamento / 40 | JSON válido / 120 |
|---|---:|---:|---:|---:|---:|---:|---:|
| arithmetic-15b, sem subtotais | 80 | 160 | 7 | 0 | 6 | 1 | 119 |
| math-prefix-15b | 198 | 400 | 7 | 1 | 5 | 1 | 95 |
| mixed-15b, sem subtotais | 96 | 192 | **57** | 1 | 30 | 26 | 118 |
| mixed-prefix-15b | 144 | 384 | 41 | 2 | 14 | 25 | 91 |

Os subtotais elevaram o acerto aritmético em **um caso por braço**; isso
equivale a apenas 1/40 e 2/40, respectivamente. O total do especialista
continuou em 7/120. No misto, o total caiu de 57/120 para **41/120**. O
principal recuo foi em dedução (30 para 14). O formato JSON estrito também
piorou substancialmente. Se avaliarmos só respostas numéricas/nomes legíveis
sem exigir evidências válidas, o misto com subtotais atinge 42/120: a
diferença de um acerto não altera a conclusão.

| Braço com subtotais | Limpo / 30 | Não relacionado / 30 | Numérico / 30 | Similar / 30 | Evidências exatas / 120 |
|---|---:|---:|---:|---:|---:|
| math-prefix-15b | 4 | 1 | 1 | 1 | 12 |
| mixed-prefix-15b | 16 | 9 | 8 | 8 | 15 |

O misto foi especialmente frágil com distratores numéricos em dedução:
0/10 respostas corretas nessa combinação, contra 7/10 em dedução limpa.
Esse contraste descreve o comportamento observado, mas o tamanho do grupo
é pequeno. As figuras individuais de [math-prefix-15b](../reports/2026-09-30/arithmetic-curriculum/math-prefix-15b/comparison-before-after.png)
e [mixed-prefix-15b](../reports/2026-09-30/arithmetic-curriculum/mixed-prefix-15b/comparison-before-after.png)
mostram as transições pareadas de cada caso.

Os 400 e 384 passos consumiram, respectivamente, 102,9 s e 114,5 s apenas
nas atualizações do otimizador, com 2.400 e 2.801 tokens supervisionados.
As perdas de *treino* chegaram perto de zero, mas isso **não** se traduziu em
generalização aritmética. O tempo de geração dos 120 casos foi 150,9 s e
177,6 s. Treino e inferência foram locais na RTX 3060 Laptop de 6 GB.

Esta comparação **não isola** o efeito de inserir subtotais: o número de
exemplos únicos e de passos também aumentou. Cada braço tem uma única
semente, e o split de validação já guiou escolhas anteriores. Os dados
apoiam uma decisão prática limitada: **não usar estes dois checkpoints como
modelo principal nem repetir a mesma estratégia com mais passos**. O melhor
ajuste local observado continua sendo `mixed-15b`, com 57/120 na validação;
uma avaliação em problemas novos e independentes é necessária para estimar
se esse ganho se mantém.

Os [artefatos desta rodada](../reports/2026-09-30/arithmetic-curriculum/)
incluem configuração, IDs e ordem exata de treino, histórico de perda,
manifestos, respostas brutas, avaliação estrita e diagnóstico de conteúdo.
Pesos e estados do otimizador permanecem em `data/` local.

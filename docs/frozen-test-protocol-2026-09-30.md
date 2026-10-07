# Protocolo congelado do teste final — 30/09/2026

Este arquivo é registrado antes de abrir o split `test` v2.1.

O conjunto final contém 120 casos: 30 problemas-base, dez por tarefa,
quatro condições pareadas por base. Usa profundidade cinco e templates
distintos de treino e validação; a mudança conjunta impede atribuir erros
somente à profundidade ou à linguagem.

Executar **uma única vez** o checkpoint original C0 e os checkpoints finais
de C1, C2 e C3 na semente de treinamento 42. A semente 42 foi fixada no
protocolo local, não escolhida por desempenho no teste. Os braços foram
selecionados após inspecionar apenas a validação. Usar revisão 1,5B já
registrada, BF16, prompt evidence-json-v1, seed de geração 42, greedy e
limite de 128 tokens. Preservar todas as saídas, inclusive falhas.

Métricas principais: acurácia exata da resposta nos 120 casos e por tarefa
e condição; transições pareadas em relação ao original. Relatar também
aceitação do contrato, seleção exata de fatos, tokens e tempo. A análise
de conteúdo aceita um JSON completo ou um único bloco Markdown; não extrai
respostas de texto livre. O mesmo parser é aplicado a todos os braços.

Não ajustar modelos ou prompts depois de ver o teste, nem selecionar a
melhor semente em retrospecto. Se o resultado for pior, preservá-lo.
As três sementes e intervalos do estudo permanecem análises de validação;
o teste final é uma checagem independente, sem intervalo honesto para
variabilidade de treinamento a partir de uma única semente por braço.

Auditoria manual anterior ao teste: foi lido o primeiro problema-base de
cada tarefa nas condições limpa e similar (seis casos). Gabaritos
conferidos com as cadeias de fatos. No exemplo aritmético `000008`,
70−2−6−8−7=47; C2/C3 erraram. Na dedução similar `000008`, o resultado
correto é P2134; ambos pararam em P1746. No acompanhamento similar
`000008`, a sequência temporal leva a B7771; C2 deu B3037 e C3 deu F7.
Isso é uma checagem qualitativa delimitada, não uma validação humana
exaustiva do benchmark.

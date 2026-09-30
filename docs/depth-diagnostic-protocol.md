# Diagnóstico local de profundidade — protocolo v1

Definido antes da execução. Objetivo: medir competência limpa em cadeias de uma,
duas, três e quatro operações, sem mudar simultaneamente o template de linguagem.

- 120 novas inferências: três tarefas × dez cadeias × quatro profundidades.
- Aritmética, dedução e acompanhamento de objetos; inglês sintético, sem ruído.
- Mesmas frases compartilhadas e mesma ordem relativa entre profundidades.
- Cadeia completa gerada primeiro; prefixos lógicos definem os níveis menores.
- Templates da validação v2; IDs e aleatoriedade em namespace diagnóstico separado.
- Qwen2.5-1.5B-Instruct, revisão fixa, BF16, prompt padrão evidence-json-v1.
- Respostas brutas, legibilidade e acurácia por tarefa/profundidade, sem consertar saídas.
- Nenhum exemplo do teste final é aberto. Estes exemplos são desenvolvimento,
  não substituem o benchmark congelado e não serão usados como resultado final.

Limitações: dez cadeias por tarefa, níveis pareados e dependentes; aumentar profundidade
também aumenta comprimento e pode mudar o gabarito. O diagnóstico não isola mecanismo
interno nem demonstra transferência ou eficácia de treinamento.

Execução: `poetry run python scripts/diagnose_depth.py data/depth-diagnostic/2026-09-29`.
Interrupções podem ser retomadas com código/configuração idênticos no mesmo destino.

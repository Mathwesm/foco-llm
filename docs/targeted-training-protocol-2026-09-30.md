# Rodada exploratória dirigida — 30/09/2026

Este protocolo é posterior ao teste final congelado e **não o reabre**.
O objetivo é investigar a falha de aritmética e tentar obter um adaptador
mais equilibrado. Todas as decisões desta rodada usam apenas os 960 exemplos
de treino e os 120 de validação v2.1; o split `test` anterior não entra em
seleção, ajuste nem reavaliação destes novos adaptadores.

Cinco braços foram definidos antes da execução, todos com condição de treino
`similar`, supervisão apenas da resposta, rank LoRA 4, taxa 1e-4, seed 42,
duas passagens sobre os exemplos e o mesmo prompt/decodificação de validação:

| Braço | Modelo | Bases de treino | Passos | Objetivo |
|---|---|---:|---:|---|
| arithmetic-15b | Qwen2.5 1,5B | 80 aritmética | 160 | Corrigir tarefa sem acertos |
| text-15b | Qwen2.5 1,5B | 40 dedução + 40 acompanhamento | 160 | Testar duas tarefas de texto |
| mixed-15b | Qwen2.5 1,5B | 32 por tarefa | 192 | Modelo único equilibrado |
| mixed-11b | TinyLlama 1,1B | 32 por tarefa | 192 | Explorar tamanho intermediário |
| mixed-05b | Qwen2.5 0,5B | 32 por tarefa | 192 | Explorar tamanho menor |

Os grupos multitarrefa são intercalados e usam bases disjuntas por tarefa.
O orçamento de exemplos e passos **não é igual entre braços especialistas e
mistos**. TinyLlama tem outra família, tokenizer e pré-treino; sua diferença
para Qwen não isola o efeito de parâmetros. Cada tamanho precisa do próprio
baseline sem ajuste no mesmo split. Registros de treino, respostas brutas e
falhas permanecem separados por braço.

O critério exploratório principal é acurácia de resposta por tarefa e
condição. Relatar também seleção exata de evidências, formato, duração e
tokens. Um acerto aritmético isolado não basta para afirmar generalização;
procurar desempenho em casos limpos e com distratores e conferir transições
pareadas. O melhor braço na validação será candidato para um **novo** teste
reservado em dados futuros, não para reinterpretação do teste já usado.

Parar uma expansão de orçamento se houver erro de infraestrutura, falta de
memória ou nenhuma melhora útil na validação; preservar resultados negativos.
Não fazer busca retroativa de hiperparâmetros sobre o teste final.

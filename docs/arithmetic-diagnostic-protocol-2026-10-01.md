# Diagnóstico aritmético local: protocolo anterior à execução

Objetivo: separar **cálculo**, **interpretação dos fatos** e **resistência a
distratores** antes de decidir outro treinamento. Esta rodada é exploratória;
não usa o teste final congelado nem altera checkpoints existentes.

O benchmark v2.1 será gerado com semente **20261002**, 100 problemas-base
por tarefa. Serão selecionadas as dez bases aritméticas da partição de
validação. Para cada base, serão construídos sete casos com o mesmo gabarito
aritmético independente em cada profundidade:

| Caso | Conteúdo apresentado | Atualizações |
|---|---|---:|
| `expression-1/2/4` | Expressão numérica explícita, sem seleção de fatos | 1, 2, 4 |
| `text-1/2/4` | Frases relevantes, sem distrator | 1, 2, 4 |
| `similar-4` | Problema v2.1 completo com distrator semelhante | 4 |

Total: **70 respostas por modelo**, dez bases × sete formas. O item
`similar-4` é pareado com `text-4`. As versões curtas são derivadas somente
para **avaliação**; não se tornam exemplos de treino. O mesmo prompt JSON,
decodificação gulosa BF16 e limite de 128 tokens serão usados em todos os
casos. Respostas ausentes, malformadas ou incorretas contam como erros;
acerto da resposta e validade de formato serão relatados separadamente.

Comparação primária: Qwen2.5 0,5B, TinyLlama 1,1B e Qwen2.5 1,5B, cada um
sem ajuste e com o adaptador misto local correspondente. Se a expressão
direta também falhar, investigar capacidade de cálculo ou formato antes de
alterar seleção de contexto. Se expressão funcionar e texto limpo falhar,
investigar a tradução dos fatos para operações. Se só o distrator derrubar
o resultado, investigar seleção de fatos. Essas regras são diagnósticas,
não inferências sobre mecanismos internos do modelo.

Os modelos têm famílias e pré-treinos distintos; a diferença entre eles
não isola o número de parâmetros. Mudanças posteriores de treino terão
protocolo e avaliação separados. Não ajustar prompt ou hiperparâmetros
com base nos primeiros resultados parciais desta matriz.

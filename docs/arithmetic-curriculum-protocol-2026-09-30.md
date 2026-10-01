# Hipótese adicional: subtotais aritméticos

Registrado depois da matriz exploratória de cinco braços e antes da nova
execução. O teste final usado anteriormente permanece fechado.

Nos 40 casos aritméticos de validação, o melhor braço da matriz obteve
4 acertos (TinyLlama 1,1B); o Qwen 1,5B misto obteve 1. Em exemplos
limpos, o modelo pode selecionar os fatos certos e errar o cálculo.
Esta rodada testa se exemplos de **subtotais progressivos** ajudam.

Para cada problema aritmético de treino selecionado, criar problemas
limpos derivados do valor inicial mais os primeiros 1, 2, ... eventos
relevantes. O gabarito de cada subtotal é calculado pelo solver
independente, e cada caso mantém somente fatos daquele prefixo. O
problema original com distratores similares continua no treino.
As perguntas mantêm o mesmo formato de resposta JSON e a avaliação usa
o mesmo prompt, parser e 120 casos de validação. Não se derivam exemplos
de validação ou teste.

| Braço | Modelo | Bases originais | Passos | Contraste exploratório |
|---|---|---:|---:|---|
| math-prefix-15b | Qwen2.5 1,5B | 80 aritméticas | 400 | arithmetic-15b, 160 passos sem subtotais |
| mixed-prefix-15b | Qwen2.5 1,5B | 32 por tarefa | 384 | mixed-15b, 192 passos sem subtotais |

Rank 4, learning rate 1e-4, seed 42, BF16, geração gulosa e limite de
128 tokens são mantidos. **Aumentam simultaneamente** exemplos únicos e
passos; portanto um eventual ganho não isola o efeito dos subtotais.
Relatar acurácia por tarefa/condição, evidências, formato e custo, inclusive
se houver regressão. A finalidade é encontrar uma direção útil de treino,
não reivindicar causalidade a partir deste único par de braços.

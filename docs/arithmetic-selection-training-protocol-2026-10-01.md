# Treino local de seleção de evidências

Registrado após a avaliação do prompt `arithmetic-selection-v2`, que
obteve 0/10 seleções exatas com distrator nos três braços existentes.
Esta é uma intervenção exploratória sobre a **tarefa de selecionar fatos**,
não um ajuste retroativo do teste final congelado.

Treinar `Qwen/Qwen2.5-1.5B-Instruct` desde a revisão base fixada, com
LoRA rank 4 em `q_proj`/`v_proj`, BF16, learning rate `1e-4`, semente 42,
batch 1 e **320 atualizações**. Selecionar 80 problemas aritméticos de
treino na condição `similar`, com a rotina determinística já usada nos
pilotos. Cada entrada usa exatamente o prompt `v2` da avaliação; a saída
supervisionada contém **somente** `{"evidence":[...]}`. A máscara de loss
cobre apenas a conclusão do assistente. A ordem dos exemplos é fixa e
repetida ciclicamente, quatro passadas pelo conjunto.

O protocolo de checkpoint por passo, integridade dos pesos, retomada e
verificação de recarga é o mesmo do piloto local. Medir perda no exemplo
de treino apenas como diagnóstico funcional, sem chamá-la de acurácia.
Depois avaliar as mesmas 40 entradas de validação do experimento `v2`,
sem usar gabaritos de validação no treino, preservando todas as respostas
brutas. Métrica principal: seleção exata nos dez casos `similar-4`;
secundárias: validade de formato, contagem de fatos relevantes omitidos e
irrelevantes incluídos, resposta do executor determinístico.

Esta comparação muda objetivo de supervisão, conjunto de exemplos e
número de passos simultaneamente. Portanto um eventual ganho não isola
qual mudança causou o efeito. Os templates sintéticos continuam limitando
generalização para texto livre.

## Extensão: seletor treinado + cálculo pelo modelo

Após observar 6/10 seleções exatas com distrator e 6/10 respostas do
executor, avaliar o sistema sem calculadora determinística. O novo
adaptador `evidence_only` fornece apenas IDs; as frases selecionadas
produzem uma expressão por extração sintática fixa, sem consultar a
evidência ou resposta de referência. Passar **somente essa expressão** ao
adaptador `math-prefix-15b`, uma operação por chamada, sempre usando seu
subtotal produzido no passo seguinte. Se a seleção não formar exatamente
um valor inicial e quatro atualizações, o caso falha. Medir o resultado
final nos mesmos dez casos `similar-4`, com dez no denominador, inclusive
falhas de seleção. O cálculo com gabarito não entra no prompt. Este é um
resultado **pós-hoc**, e não um novo teste final.

## Controle sintático pós-hoc

Como as frases artificiais nomeiam explicitamente o recipiente-alvo,
avaliar também um seletor sem aprendizado: extrair o nome da pergunta e
selecionar toda frase que contém exatamente esse nome como palavra. O
seletor recebe apenas pergunta e fatos visíveis; o executor permanece
igual. Esse controle serve para identificar se o benchmark atual já é
resolvido por uma regra lexical, e não para reivindicar generalização a
textos naturais com correferência ou sinônimos. Registrar este controle
como **pós-hoc**, pois foi proposto depois de ver o ganho do novo treino.

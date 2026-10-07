# Seleção de fatos antes do cálculo: protocolo exploratório

Esta rodada responde à pendência do diagnóstico aritmético: o modelo deve
identificar os fatos relevantes **sem receber os IDs corretos**. A fonte é o
mesmo conjunto de dez bases de validação, semente `20261002`. Para cada
base, usar `text-1`, `text-2`, `text-4` e `similar-4`: quarenta casos por
modelo. A condição `similar-4` tem cinco fatos relevantes e cinco fatos
de outro recipiente. O teste final congelado permanece fechado.

O prompt novo pede apenas um array JSON com os IDs do valor inicial e das
atualizações que respondem à pergunta. O alvo é extraído da **pergunta
visível**; não se usa o campo `evidence` no prompt. Aceitar somente um
array completo, IDs existentes, únicos. Comparar os IDs selecionados com
os de referência. Em seguida, um executor determinístico lerá **apenas
os fatos selecionados**, fará a soma/subtração e comparará com o gabarito.
O executor não usa o alvo, a evidência ou a resposta de referência para
decidir o cálculo. Se a seleção incluir dois valores iniciais, nenhum
valor inicial ou fatos malformados, o caso falha em vez de ser reparado.

Relatar validade do formato, seleção exata, precisão/recall dos fatos,
resposta do executor e acerto final, por condição e modelo. Isso mede a
capacidade de seleção separada da aritmética do LLM; um executor correto
não constitui melhora dos pesos do modelo. Comparar pelo menos o Qwen
1,5B original, o ajuste misto e o ajuste de subtotais, mantendo BF16,
geração gulosa, revisões e adaptadores fixados. As dez bases são pequenas
e sintéticas; a rodada é exploratória e não será usada para escolher um
hiperparâmetro com alegação confirmatória.

## Extensão de formato registrada após a primeira execução

No primeiro braço `math-prefix-15b`, o prompt de array (`v1`) gerou 0/40
arrays de IDs válidos. As saídas eram listas de **quantidades numéricas**,
por exemplo `[44]`, mesmo com a instrução para retornar IDs. Por isso
faremos uma segunda rodada `v2`, preservando a primeira: um exemplo curto
de seleção mostra o objeto `{"evidence":["F1","F3"]}`, sem resposta
numérica. O parser só aceita esse objeto completo, com IDs válidos. Essa
modificação explora uma incompatibilidade de formato observada e não será
misturada com a rodada `v1` nem tratada como teste confirmatório.

# Diagnóstico local da aritmética — 1º de outubro de 2026

## Pergunta e desenho

Investigamos se os erros aritméticos vêm principalmente do contrato de
saída JSON, da leitura dos fatos, da escolha dos fatos relevantes ou da
manutenção de subtotais. Os protocolos foram registrados antes de cada
rodada em [diagnóstico](arithmetic-diagnostic-protocol-2026-10-01.md),
[controle matemático](math-specialist-control-protocol-2026-10-01.md) e
[decomposição](chained-arithmetic-protocol-2026-10-01.md). As respostas
brutas, manifests, pontuações e a [figura](../reports/2026-10-01/arithmetic-diagnostic/arithmetic-depth.png)
estão em `reports/2026-10-01/arithmetic-diagnostic/`.

Foram geradas dez bases **aritméticas de validação** com a semente
`20261002`, independentes das bases usadas no treino. Cada base originou
sete formas: expressão direta, fatos limpos, ambas com 1, 2 e 4 operações,
e texto com distrator semelhante com 4 operações. Portanto são 70 casos
por modelo, dez por célula; não são 70 perguntas independentes. A
comparação usa o gabarito do gerador/solver, inferência gulosa BF16 e
modelo/revisão fixados nos manifests. Máquina: RTX 3060 Laptop GPU. Esta
rodada **não abriu o teste final congelado**.

## Resultado por profundidade e contexto

Cada número é acertos em dez casos. `E` é expressão numérica, `T` é texto
somente com fatos relevantes e `D4` inclui distrator semelhante.

| Modelo local | E1 | E2 | E4 | T1 | T2 | T4 | D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 0,5B original | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwen 0,5B misto | 1 | 0 | 0 | 2 | 0 | 0 | 0 |
| TinyLlama 1,1B original | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TinyLlama 1,1B misto | 1 | 0 | 0 | 5 | 0 | 0 | 0 |
| Qwen 1,5B original | 6 | 3 | 2 | 4 | 0 | 0 | 0 |
| Qwen 1,5B misto | 7 | 4 | 2 | 6 | 2 | 2 | 0 |
| Qwen 1,5B aritmética | 7 | 4 | 0 | 7 | 1 | 1 | 0 |
| Qwen 1,5B texto | 7 | 2 | 2 | 3 | 0 | 0 | 0 |
| Qwen 1,5B subtotais | **10** | **6** | 1 | 8 | 2 | 1 | 0 |
| Qwen 1,5B misto + subtotais | 9 | 3 | 1 | **9** | 2 | 0 | 0 |

O Qwen 1,5B original produziu respostas legíveis em 58/70, mas **nenhuma**
no JSON estrito; o misto produziu 70/70 no contrato. O TinyLlama original
produziu 0/70 respostas legíveis no parser, então seus zeros não medem
isoladamente capacidade aritmética. O treinamento de subtotais elevou E1
para 10/10, mas E4 ficou em 1/10. Em todos os braços, D4 foi 0/10.

Isso aponta para **mais de uma limitação**: a conta já falha com a expressão
pronta; o texto e o distrator acrescentam perdas. Não há evidência aqui
para atribuir a causa a um hiperparâmetro específico, ao número de
parâmetros isoladamente ou a um mecanismo interno de atenção.

## Formato simples e controle matemático

Removemos o contrato JSON/evidências e pedimos só o inteiro, nas mesmas
30 expressões (dez de cada profundidade):

| Qwen 1,5B | 1 operação | 2 operações | 4 operações |
|---|---:|---:|---:|
| Original | 7/10 | 5/10 | 3/10 |
| Misto | 9/10 | 4/10 | 2/10 |
| Subtotais | **10/10** | 5/10 | **0/10** |

Quase todas as saídas obedeceram ao formato de inteiro, inclusive 10/10
nas expressões de quatro operações do modelo de subtotais. Assim, o JSON
não explica sozinho os erros. Este é um prompt pós-hoc exploratório,
não a métrica principal do projeto.

Também rodamos o Qwen2.5-Math-1.5B-Instruct fixado à revisão registrada
no manifest. No pedido de **inteiro sem raciocínio**, produziu 0/30 saídas
válidas porque seguiu seu estilo de raciocínio longo; isso **não equivale a
0/30 em matemática**. No prompt recomendado de raciocínio e resposta
`\boxed{}`, com limite de 512 tokens, obteve 8/10, 5/10 e 0/10. Em uma
resposta de quatro operações, o modelo trocou `+ 8` por `- 8` ao reescrever
a própria expressão; em outras, a cópia era correta, mas um subtotal
estava errado. O controle reforça a necessidade de medir transcrição e
cálculo separadamente, sem tratar o modelo matemático como comparável
diretamente no mesmo prompt.

## Intervenção: quatro chamadas curtas

Para as mesmas dez expressões de quatro operações, um parser separou a
expressão **já fornecida**. A cada chamada o modelo recebeu só uma soma
ou subtração e o subtotal **que ele mesmo produziu** seguiu para a próxima.
Um erro, portanto, não foi corrigido com o gabarito. `operation_correct`
mede a operação sobre o subtotal recebido; `subtotal_correct` mede a
igualdade com o subtotal verdadeiro. Os relatórios definitivos desta
intervenção são os diretórios `chained-*-v2`.

| Qwen 1,5B | Expressão inteira, só número | Quatro chamadas, resposta final |
|---|---:|---:|
| Original | 3/10 | 0/10 |
| Misto | 2/10 | 4/10 |
| Subtotais | 0/10 | **9/10** |

No modelo de subtotais, as operações locais tiveram 10/10, 10/10, 9/10
e 10/10 acertos por posição. O único erro foi `84 + 5 → 90` (correto:
`89`); a quarta conta `90 + 1 → 91` foi localmente correta, mas herdou o
subtotal errado, deixando o resultado final `91` em vez de `90`.

**Leitura prática:** há uma solução local eficaz para aritmética de
expressões explícitas neste pequeno conjunto: usar o adaptador de
subtotais com decomposição externa. Ela custa quatro chamadas de geração
por problema e **não demonstra** que o modelo reconhece sozinho os fatos
relevantes em texto. D4 permaneceu 0/10 na avaliação de chamada única;
a decomposição ainda não foi avaliada depois de uma seleção de fatos feita
pelo próprio modelo. Para um sistema que precisa de exatidão matemática,
um executor determinístico para a operação escolhida é a opção mais
confiável; a parte de pesquisa permanece identificar corretamente **qual**
operação aplicar e quais fatos ignorar.

## Limites e próximo experimento

Cada célula tem apenas dez casos correlacionados por base. O treino de
subtotais também mudou quantidade de exemplos e passos, portanto o ganho
não isola a técnica. O controle matemático usa prompt diferente. Não
generalizar percentuais para textos livres ou alegar efeito causal dos
parâmetros. Para continuar o TCC, testar seleção de fatos sem gabarito,
registrar expressão extraída, acerto da extração e acerto do cálculo em
etapas distintas, e repetir em mais sementes antes da comparação em GPU
de nuvem.

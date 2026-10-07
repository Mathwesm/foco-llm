# Verificação local em novas instâncias sintéticas

O [protocolo](fresh-instance-protocol-2026-09-30.md) fixou os modelos e a
semente **20261001** antes da geração. O novo dataset tem SHA-256
`8d2d5b5ffbf1c3bdb07c6e79960d5744582b5608c05bc3dc37ba73401d590d75`.
Foram avaliados os 120 exemplos da partição de validação: 40 de aritmética,
40 de dedução e 40 de acompanhamento de objetos, cada tarefa com dez casos
em contexto limpo e dez em cada tipo de distrator. Nenhum dos 120 registros
é idêntico ao da validação anterior, embora os templates do gerador sejam
os mesmos. O teste final antigo não foi usado.

| Modelo | Acertos estritos / 120 | Respostas legíveis corretas / 120 | Aritmética / 40 | Dedução / 40 | Acompanhamento / 40 | JSON válido / 120 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 1,5B original | 0 | 8 | 0 | 7 | 1 | 0 |
| Qwen2.5 1,5B `mixed-15b` | **55** | **55** | 0 | 33 | 22 | 115 |

O modelo original devolveu sobretudo blocos Markdown, por isso obteve zero
na pontuação JSON estrita. A análise de conteúdo pós-hoc, aplicada às
**mesmas respostas brutas**, recuperou oito respostas corretas; não foram
feitas novas gerações nem reparos de resposta. O adaptador alcançou 55/120
nos dois critérios. Esse valor é próximo aos 57/120 da validação usada na
escolha do modelo, mas representa apenas **uma nova semente** do mesmo
gerador. Os acertos aritméticos passaram de 1/40 na validação anterior para
**0/40** nas novas instâncias: não há evidência de competência aritmética
robusta nessa escala.

| Modelo e critério de resposta legível | Limpo / 30 | Não relacionado / 30 | Numérico / 30 | Similar / 30 | Evidências exatas / 120 |
|---|---:|---:|---:|---:|---:|
| Original | 6 | 0 | 0 | 2 | 0 |
| `mixed-15b` | 19 | 15 | 14 | 7 | 18 |

O distrator similar foi o mais difícil para o adaptador. Os 18 acertos de
evidências exatas são uma medida separada do acerto de resposta: o modelo
às vezes responde corretamente com uma cadeia de evidências incompleta ou
errada. A figura [antes/depois](../reports/2026-09-30/fresh-instances/comparison/before-after.png)
mostra acertos por tarefa; as figuras de
[conteúdo original](../reports/2026-09-30/fresh-instances/content-baseline/content-metrics.png)
e [conteúdo ajustado](../reports/2026-09-30/fresh-instances/content-mixed/content-metrics.png)
mostram as condições e a medida de evidências.

Ambos os modelos usaram a mesma revisão base, prompt `evidence-json-v1`,
BF16, geração gulosa e limite de 128 tokens. O adaptador foi carregado
apenas após conferir a integridade do checkpoint e o SHA-256 do dataset
**original de treino**; o manifesto da inferência registra separadamente o
hash do dataset novo. O conjunto publicado em
[artefatos](../reports/2026-09-30/fresh-instances/) contém o manifesto do
gerador, casos avaliados, respostas brutas, previsões, pontuação estrita,
diagnóstico de conteúdo e transições pareadas. Os pesos ficam em `data/`
local e não são publicados.

Esta rodada melhora a confiança de que o ganho em dedução e acompanhamento
não veio apenas dos 120 casos usados para escolher o braço de treino. Ainda
não permite afirmar generalização para tarefas não sintéticas, outro idioma
ou modelos maiores. O caminho local chegou a um ponto de retorno baixo:
mais passos no currículo de subtotais reduziram o desempenho, enquanto a
falha aritmética persistiu. A próxima comparação útil requer um modelo-base
mais capaz ou um método de treino aritmético diferente, em protocolo próprio.

# Seleção de fatos antes do cálculo — resultado local exploratório

O [protocolo](arithmetic-selection-protocol-2026-10-01.md) usa dez bases
aritméticas de validação em quatro formas: texto com 1, 2 ou 4 atualizações
relevantes e texto com quatro atualizações mais cinco fatos de outro
recipiente. Cada célula abaixo tem dez casos; respostas e manifests estão
em `reports/2026-10-01/arithmetic-selection/`.

O primeiro prompt (`v1`) falhou no formato em 40/40 casos do adaptador de
subtotais: devolveu listas de **números**, não IDs. A versão `v2` incluiu
um exemplo independente da validação e pediu `{"evidence":["F1",...]}`.
O executor usou somente os fatos selecionados pelo modelo. `Seleção`
significa conjunto de IDs exatamente correto; `resultado` significa
resposta correta depois de executar os fatos selecionados. Não houve
correção da seleção com o gabarito.

| Qwen 1,5B, prompt v2 | Texto 1: seleção/resultado | Texto 2 | Texto 4 | Distrator 4 |
|---|---:|---:|---:|---:|
| Original | 1/1 | 2/2 | 1/1 | **0/0** |
| Misto | 2/2 | 3/3 | 1/1 | **0/0** |
| Subtotais | 3/3 | 4/5 | 2/2 | **0/0** |

O par da última linha em `Texto 2` difere porque uma seleção incompleta
produziu o resultado certo por coincidência; por isso a seleção exata é
uma métrica separada. Na condição com distrator, as saídas tinham formato
válido em 10/10 casos por modelo, mas faltavam entre 22 e 24 dos 50 fatos
relevantes no conjunto de dez casos. Os modelos também incluíram de 5 a 8
fatos irrelevantes. O erro não é só aritmético nem só de formatação: a
seleção dos fatos ainda falha.

Esses números são pequenos e pós-hoc. O novo prompt `v2` foi desenhado
após observar o erro de formato de `v1`; portanto não é um teste final
confirmatório. O próximo braço treinará diretamente os IDs de evidência
nos exemplos **de treino**, mantendo a mesma validação para medir se a
seleção com distratores melhora. O teste final congelado continua fechado.

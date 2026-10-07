# Fechamento do TCC até 29 de outubro de 2026

Este plano usa os resultados preservados até 7 de outubro. O manuscrito em
inglês e sua versão em português ficam fora deste repositório, no diretório
local de documentos do TCC. Nenhum resultado ainda em execução entra como
concluído.

| Prazo interno | Entrega verificável | Critério de fechamento |
| --- | --- | --- |
| 10/10 | Repetição 7B com semente 43 no Kaggle | Versão salva, manifesto, respostas brutas, pontuações e resumo baixados; falhas documentadas se ocorrerem. |
| 17/10 | Análise científica congelada | Comparações por problema-base, auditoria de erros, controles simbólico e lexical, variação entre sementes e limites das inferências conferidos contra os arquivos de resultado. |
| 24/10 | Artigos em inglês e português revisados | Mesmo conjunto de números nas duas versões; figuras legíveis, exemplos das três tarefas, referências verificadas e identificação dos autores confirmada. |
| 29/10 | Entrega final | Arquivos no formato pedido pela instituição, revisão do orientador incorporada e PDF final conferido página por página. |

O teste aritmético com 60 bases e o teste entre tarefas com outras 60 bases são
os principais resultados reservados. A ablação de 20 bases e as repetições de
semente feitas depois da inspeção desses testes são análises **pós-hoc**. Essa
distinção deve aparecer também em qualquer resumo ou apresentação.

Não é necessário prolongar o treinamento por princípio. A decisão de fazer
outra execução depende da repetição 7B: se ela reproduzir a melhora dentro da
aritmética, priorizamos revisão do texto e dos artefatos; se divergir muito,
investigamos primeiro ambiente, checkpoint, prompt e dados antes de atribuir a
diferença à semente. O estudo não mede diretamente ativação, atenção causal ou
"pensamento" interno do modelo. Sua conclusão principal é comportamental:
seleção de evidências sob distratores e transferência limitada para outras
tarefas neste protocolo sintético.

Pendências externas: retorno do orientador, nomes completos e afiliações de
todos os autores, e confirmação do formato de submissão. A compilação pelo
editor LaTeX embutido apresentou erro de ambiente (`Unable to find standard
directories for platform`) em 7/10; ela precisa voltar a funcionar para a
verificação visual final, mas esse erro não prova falha no conteúdo do `.tex`.

Em 7/10, a [repetição 7B com semente 43](https://www.kaggle.com/code/matheusdsousaribeiro/foco-llm-qwen-7b-seed-43-replication)
foi salva como versão 2 e estava em execução na última verificação. A versão 1
parou antes do treino porque o ambiente padrão do Kaggle havia mudado para
Python 3.13 e o notebook exigia 3.12. A versão 2 aceita ambas as versões
suportadas pelo projeto, concluiu a instalação via Poetry e carregou os pesos
do modelo. Ainda não há resultado experimental dessa repetição.

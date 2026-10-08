# Intervenção em notas escritas pelo modelo

O objetivo é medir se uma nota intermediária escrita pelo próprio modelo
influencia a escolha posterior dos fatos. O experimento é inspirado na remoção
de trechos de raciocínio de Chen et al., *Extracting Search Trees from LLM
Reasoning Traces Reveals Myopic Planning* (arXiv:2605.06840v5), mas **não é uma
replicação**: nosso modelo Qwen2.5-Instruct não expõe um canal interno de
raciocínio. A nota é gerada por uma solicitação explícita e reapresentada ao
modelo como texto no prompt seguinte.

O protocolo usa as primeiras dez bases de dedução e dez de rastreamento com
distrator de cadeia semelhante do holdout de 8 de outubro, SHA-256
`5a1ee80d4a2af8c27b50fc5e6fbde6eb2369388e01a21c298093447181d64951`.
Cada caso recebe uma nota com uma linha por fato visível, na ordem original,
seguida de um JSON final. O JSON final é descartado antes da segunda consulta.
Somente notas com uma linha inequívoca por fato entram na comparação. O texto
bruto das rejeições é preservado em `rejected.json`.

A segunda consulta mantém **todos os fatos e a pergunta**. Quatro braços usam
o mesmo modelo e a mesma geração gulosa: sem nota, nota completa, nota sem a
primeira linha de fato relevante e nota sem a primeira linha de fato
irrelevante. A escolha das linhas a remover usa o gabarito somente no programa
experimental, nunca no prompt do modelo. As métricas são validade do JSON,
seleção exata e mudança pareada dos IDs escolhidos em relação à nota completa.
As 20 bases, e não as quatro respostas por base, são as unidades pareadas.

O piloto local com Qwen2.5-1.5B-Instruct foi interrompido após 16 notas, todas
inelegíveis: algumas eram incompletas e outras repetiam IDs até o limite de
tokens. Isso é uma falha de formato da *nota*, não uma estimativa da influência
causal da nota na resposta. Os checkpoints brutos estão em
`data/reasoning-note-intervention/local-base/0572e1434da9601d/`, ignorado
pelo Git. O notebook `notebooks/kaggle_reasoning_note_intervention_7b.ipynb`
prepara o mesmo conjunto para o Qwen2.5-7B-Instruct no Kaggle e grava
`comparison.json`, `summary.json`, `scores.json` e `rejected.json`.

A execução salva do 7B também não produziu notas elegíveis: **0/20**.
[Versão executada no Kaggle](https://www.kaggle.com/code/matheusdsousaribeiro/kaggle-reasoning-note-intervention-7b)
(versão 2, fonte `94ae2fafabfa12a21bc2c1d5cf11c6212dd9ff8b`). Assim,
as quatro intervenções não têm comparação pareada interpretável; células
com zero no resumo não são acurácia de 0%. Uma eventual simplificação do
formato será registrada como outro protocolo, com novos dados e denominador.

Mesmo quando válido, remover uma linha altera o contexto fornecido ao modelo;
isso demonstra sensibilidade a texto externo produzido por ele, não revela
ativações nem prova um mecanismo interno de pensamento. A ordem de remoção é a
ordem de apresentação dos fatos, não a profundidade lógica da cadeia. Como o
teste foi proposto após resultados anteriores, qualquer achado é exploratório.

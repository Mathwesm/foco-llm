# Diagnóstico local com paráfrases e distrator lexical

Este teste **pós-hoc e exploratório** usa as dez bases de validação da
condição `similar-4` que já haviam sido examinadas. Não houve novo treino.
Para cada base, o roteiro preservou os números, as operações, a pergunta,
a resposta de referência e os IDs dos cinco fatos necessários. Reescreveu
os fatos com linguagem diferente: o valor inicial passou a usar
`At the start, ... held ... marbles`; as quatro atualizações do recipiente
perguntado passaram a dizer `the container named in the question` em vez
de repetir seu nome. Os fatos do outro recipiente também foram
reescritos. Acrescentou-se um fato irrelevante que **menciona o nome do
recipiente perguntado** e contém um número, como `The painted label on
copper shows the number 27.`

Essa transformação testa duas fragilidades específicas: ligar uma
expressão à pergunta e rejeitar um número irrelevante apesar da
correspondência literal do nome. A regra lexical anterior falha por
construção nesse desenho; isso é um controle de sanidade da perturbação,
não uma vitória demonstrada do modelo. A seleção é avaliada sem passar
as novas frases ao executor aritmético sintático, que foi escrito para
os templates originais.

| Braço | Seleção exata | Saída JSON válida | Fatos relevantes recuperados | Irrelevantes incluídos |
|---|---:|---:|---:|---:|
| Regra lexical | 0/10 | 10/10 | 10/50 | 10 |
| Qwen 1,5B original | 0/10 | 9/10 | 13/50 | 6 |
| Qwen 1,5B com adaptador de seleção | 0/10 | 9/10 | 20/50 | 16 |

O adaptador recuperou sete fatos necessários a mais que o modelo
original, mas adicionou dez irrelevantes a mais. **Nenhum resolveu por
completo um caso.** Em comparação, o adaptador havia selecionado
exatamente 6/10 casos no `similar-4` original. Isso sugere dependência
forte da forma textual dos exemplos de treino; não mede, isoladamente,
uma capacidade geral de entender paráfrases. São apenas dez bases
reutilizadas, com uma única transformação artificial e escolhida depois
de ver os resultados anteriores.

O [artefato integral](../reports/2026-10-01/paraphrase-stress/study.json)
contém os dez textos, rótulos, respostas brutas de cada braço, erros
por caso, versões do ambiente e hashes da base e do adaptador. Para
reexecutar localmente com o mesmo checkpoint:

```powershell
poetry run python scripts/run_paraphrase_stress.py
```

Conclusão prática para o TCC: reportar o ganho no benchmark original
com a ressalva de que ele **não se manteve** sob este pequeno teste de
paráfrase. Um estudo de generalização exigiria mais textos, estilos e
sementes definidos antes da avaliação, idealmente anotados sem usar os
mesmos dez casos explorados aqui.

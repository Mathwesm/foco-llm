# Falha numérica na tentativa de 1,5B em FP16

A execução `f0d3988e46770dcb` concluiu 120 exemplos de validação, mas produziu apenas o caractere `!` repetido. Cada saída atingiu 128 tokens, totalizando 15.360 tokens e 584,24 segundos de geração. Não interpretar os zeros do avaliador como medida de capacidade do modelo.

O primeiro prompt de validação foi investigado diretamente: os 151.936 logits do próximo token eram NaN em FP16. Ao converter o modelo carregado para BF16, os logits ficaram finitos. Esse diagnóstico isolado não determina a causa exata da instabilidade nem garante estabilidade em qualquer tarefa. Os baselines novos carregam BF16 diretamente do checkpoint, em vez de reutilizar a conversão usada no diagnóstico.

Foi adicionada uma proteção no caminho de geração que interrompe com `NumericalInferenceError` ao detectar NaN, infinito positivo ou uma sequência sem nenhum token candidato finito. Máscaras de infinito negativo continuam permitidas quando existe candidato válido. Não há correção silenciosa de scores. A falha ocorre antes de publicar uma resposta como checkpoint concluído.

O teste real confirmou: FP16 acionou a proteção; BF16 não acionou e gerou o início de um objeto JSON. A prévia é limitada a oito tokens e **não é avaliada como resposta correta**. O [registro do diagnóstico](../reports/2026-09-27/precision-diagnostic/results.json) e o script abaixo permitem verificar o procedimento.

```powershell
poetry run python scripts/check_precision.py data/processed/2026-09-27/969659029e986d86/dataset.json data/diagnostics/precision/results.json
```

Os testes sem GPU cobrem NaN parcial/total, infinito positivo, linhas totalmente mascaradas e preservação de scores válidos. O teste local com o backend real complementa esses testes. **Atualização:** as duas execuções completas em BF16 concluíram 120 exemplos cada, com a proteção ativa, sem falha numérica. [Comparação válida, métricas e limitações](model-comparison-2026-09-27.md). Isso confirma estabilidade somente nos prompts e na configuração avaliados.

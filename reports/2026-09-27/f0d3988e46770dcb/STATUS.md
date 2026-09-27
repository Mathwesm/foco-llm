# Execução inválida para comparação de competência

Todas as 120 saídas repetem `!` até o limite de tokens. O diagnóstico encontrou NaN nos logits em FP16 para o primeiro prompt de validação. Os arquivos foram preservados para auditoria da falha, e suas pontuações zero **não devem ser usadas para comparar capacidade entre modelos**.

Ver [diagnóstico numérico](../../../docs/numerical-failure-2026-09-27.md). A comparação válida será refeita com os dois modelos em BF16 e proteção contra logits inválidos.

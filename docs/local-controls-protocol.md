# Controles locais — protocolo de 30/09/2026

Após o piloto exploratório, congelar nove execuções: C1, C2 e C3, cada uma
com seeds de treinamento 42, 43 e 44. Não alterar configurações durante o lote.

| Braço | Contexto de treino | Tokens que recebem perda |
|---|---|---|
| C1 | Limpo | Prefixo JSON da resposta |
| C2 | Distrator similar | Prefixo JSON da resposta |
| C3 | Distrator similar | Resposta, evidências e encerramento |

Mesmos 32 problemas-base de dedução, selecionados com seed de seleção 42,
mesma ordem e 64 passos (duas passagens); seed de treino altera inicialização
dos adaptadores. Usar o modelo, LoRA e learning rate do piloto. A condição C3
é repetida no novo protocolo, sem misturar o piloto anterior como repetição.

O JSON completo é idêntico entre C2/C3. Para C1/C2, máscara por offsets do
tokenizador retém somente tokens integralmente contidos no prefixo da resposta,
antes do campo de evidências. Tokens que cruzam a fronteira também são
ignorados. As evidências aparecem depois da resposta; a atenção causal impede
que influenciem a previsão dos tokens anteriores. Prompt e sufixo têm loss -100.
O executor verifica igualdade dos IDs entre a tokenização com offsets e a
tokenização original; divergência interrompe o treino, sem truncamento.

O contraste C2/C3 varia também a supervisão do encerramento e o número de
tokens com perda. Portanto não isola perfeitamente a evidência. Registrar
tokens supervisionados e limitar as conclusões; não alegar orçamentos de tokens
iguais quando somente exemplos e passos são iguais.

Avaliar 120 variantes dos mesmos 30 problemas-base de validação, com seed de
inferência 42, geração gulosa e os controles do baseline. Aritmética e
acompanhamento não entram no ajuste. O teste final continua reservado.

Resultados negativos são preservados. Reamostrar por problema-base, mantendo
variantes e braços juntos, para intervalos exploratórios; três sementes não
garantem precisão estatística. Relatar também todos os resultados individuais.

Execução local sequencial, subprocessos isolados, timeout de 30 minutos por
etapa, logs em arquivo, checkpoint a cada atualização e marcador por braço
concluído. Falha encerra com erro e preserva artefatos. Sem serviço pago ou
notificação externa; acompanhamento manual nesta sessão. Repetir o comando
retoma somente se código, configuração e dataset forem compatíveis.

```powershell
poetry run python scripts/run_local_controls.py data/benchmark-v21/2026-09-28/b142fe60ba9cd6a5/dataset.json reports/2026-09-28/145ba1dd1550430d configs/pilot-deduction-c3.json data/local-controls/2026-09-30
```

Conclusão deste lote não equivale ao TCC completo: ainda requer auditoria dos
resultados, relatório consolidado e delimitação das comparações em nuvem.

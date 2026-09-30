# Piloto local de tarefa-fonte — 30/09/2026

Definido antes de treinar. Esta etapa amplia o teste funcional, sem substituir
a matriz científica completa. Tarefa-fonte: dedução, escolhida após diagnóstico
de competência limpa. Avaliar também aritmética e acompanhamento, excluídos do
treinamento. A escolha pós-diagnóstico torna o piloto exploratório.

Configuração: Qwen2.5-1.5B-Instruct na revisão já fixada, BF16, LoRA rank 4,
alpha 8, q/v, learning rate 0,0001, seed 42, batch 1, 32 exemplos diferentes,
64 atualizações (duas passagens em ordem determinística). Usar somente treino
v2.1, condição similar, supervisão de resposta e evidências (C3). Registrar
tokens com perda; não truncar sequências. Checkpoint íntegro em cada passo.

O prompt permanece igual ao baseline. A avaliação usa os 120 exemplos de
validação v2.1, nas três tarefas e quatro condições, com o mesmo parser e
geração. Não usar o teste final, selecionar checkpoint pelo melhor resultado,
ou escolher apenas exemplos acertados. Preservar também pioras.

Ainda faltam C1/C2 (resposta sem supervisão de evidências), três sementes e
intervalos de confiança. O executor deste piloto suporta tarefa, condição,
seed, orçamento e hiperparâmetros por JSON; a supervisão é fixada em C3 nesta
etapa. C1/C2 exigem máscara de perda própria antes de comparação válida.

Nenhuma conclusão sobre transferência ou mecanismo será baseada só na loss.
Relatar acurácia antes/depois por tarefa e condição e verificar os pesos-base
congelados e a recarga do adaptador.

Configuração versionada: `configs/pilot-deduction-c3.json`.

```powershell
poetry run python -m foco_llm.train_pilot data/benchmark-v21/2026-09-28/b142fe60ba9cd6a5/dataset.json configs/pilot-deduction-c3.json data/pilot-training/2026-09-30
```

A duração registrada dos passos não inclui carregamento, salvamento de checkpoints
ou verificação dos pesos. Guardar todos os checkpoints neste piloto curto; não
apagar intermediários automaticamente. Execução manual, sem serviço de nuvem.

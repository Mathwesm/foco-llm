# Conclusão local antes da ampliação em nuvem

Atualização: 29/09/2026. Não declarar a fase local concluída somente porque o smoke test passou.

## Concluído

- Dados sintéticos com separação por problema-base, gabaritos verificados e variantes pareadas.
- Inferência local com proveniência, respostas brutas e retomada.
- Comparação de dois tamanhos no piloto inicial; baseline do 1,5B na versão revisada.
- LoRA funcional, integridade de checkpoints, retomada e recarga.
- Validação após seis atualizações: sem ganho de acurácia.
- Diagnóstico de ordem; documentação e figuras.

## Em execução ou pendente

- Diagnóstico de formato concluído: original 3/30, JSON explícito 4/30, resposta simples 3/30; aritmética e acompanhamento seguem em zero. [Resultado](format-diagnostic-2026-09-29.md). Não alterado o prompt padrão.
- Dificuldade/profundidade concluída: 120 casos novos com linguagem fixa; queda de acurácia com mais etapas, sem treino. [Resultado](depth-diagnostic-2026-09-29.md). O benchmark congelado não foi modificado.
- Congelar dados, prompts, métricas e orçamento dos controles científicos.
- Executor configurável além dos seis passos funcionais, com tarefa-fonte, supervisão, sementes e validação por lote/época previamente definidos.
- C0 original, C1 treino limpo/resposta, C2 ruído/resposta, C3 ruído/resposta+evidências; controlar diferenças de formato e registrar tokens supervisionados.
- Três sementes por braço ajustado e medição de tempo/memória antes da matriz completa.
- Transferência para tarefas excluídas do treino; manter teste final reservado até congelar o protocolo.
- Intervalos por reamostragem de problemas-base, relatório de transições, curvas de treino e matriz fonte/alvo.
- Alertas de falha para execuções longas, configuração portátil e teste de instalação sem GPU.

Os itens acima não são automaticamente concluídos pela execução dos diagnósticos. O orçamento do treinamento científico depende de medições locais; não há promessa de melhora.

## Créditos AWS e preparação da nuvem

A oferta padrão atual para novos clientes anuncia até US$ 200 em créditos elegíveis, não US$ 500 universais. Uma oferta de US$ 500 pode ter regras próprias. Confirmar saldo efetivo, validade, serviços cobertos, plano da conta e região antes de dimensionar horas de GPU.

Créditos elegíveis podem cobrir EC2. Entretanto, crédito não concede cota: a documentação informa cota padrão zero para instâncias On-Demand G/VT e P, ajustável mediante solicitação. O plano gratuito possui restrições; não presumir que a GPU está disponível nem mudar para plano pago sem autorização explícita. No plano pago, consumo não coberto por créditos pode gerar cobrança.

Primeira implantação proposta: uma única máquina com GPU, teste curto, checkpoint e persistência externa, desligamento ao terminar/falhar e limite de duração. Orçamento/alerta não equivale, por si só, a bloqueio automático de cobrança. Não foram criados recursos, alteradas cotas ou acessada a conta AWS nesta etapa.

Fontes oficiais consultadas em 29/09/2026:
- https://aws.amazon.com/free/free-tier-faqs/
- https://aws.amazon.com/ec2/
- https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html
- https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html

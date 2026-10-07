# Conclusão local antes da ampliação em nuvem

Atualização: 30/09/2026. Não declarar a fase local concluída somente porque o smoke test ou um piloto passou.

**Estado atual:** a rodada local acordada foi concluída após os nove treinos,
auditoria manual delimitada, protocolo congelado e avaliação final dos quatro
braços nos 120 casos reservados. [Resultados finais](final-local-results-2026-09-30.md).
Os itens históricos abaixo registram o plano original; pendências opcionais
não impedem a passagem para Kaggle.

## Concluído

- Dados sintéticos com separação por problema-base, gabaritos verificados e variantes pareadas.
- Inferência local com proveniência, respostas brutas e retomada.
- Comparação de dois tamanhos no piloto inicial; baseline do 1,5B na versão revisada.
- LoRA funcional, integridade de checkpoints, retomada e recarga.
- Validação após seis atualizações: sem ganho de acurácia.
- Controles C1/C2/C3 concluídos com três sementes e validação completa; [resultados, respostas brutas e intervalos exploratórios](local-controls-2026-09-30.md).
- Diagnóstico de ordem; documentação e figuras.

## Histórico e melhorias opcionais

- Diagnóstico de formato concluído: original 3/30, JSON explícito 4/30, resposta simples 3/30; aritmética e acompanhamento seguem em zero. [Resultado](format-diagnostic-2026-09-29.md). Não alterado o prompt padrão.
- Dificuldade/profundidade concluída: 120 casos novos com linguagem fixa; queda de acurácia com mais etapas, sem treino. [Resultado](depth-diagnostic-2026-09-29.md). O benchmark congelado não foi modificado.
- Dados, prompts, métricas e orçamento dos nove controles congelados antes do lote; eventual controle de tokens adicional deve ser análise posterior.
- Executor de piloto C3 concluído: tarefa-fonte, condição, seed e orçamento por arquivo; 64 passos em 32 exemplos, avaliação fixa ao final. [Resultado: 6/120 → 19/120](source-task-pilot-2026-09-30.md). Faltam seleção do objetivo C1/C2 e os demais controles.
- C0 original, C1 treino limpo/resposta, C2 ruído/resposta, C3 ruído/resposta+evidências: nove execuções concluídas, diferenças de formato e tokens supervisionados registradas. Falta controle adicional de orçamento de tokens, se viável.
- Três sementes por braço ajustado concluídas; duração dos passos, memória e resultados preservados.
- Transferência para tarefas excluídas do treino; manter teste final reservado até congelar o protocolo.
- Intervalos exploratórios por reamostragem de problemas-base, transições e curvas concluídos. Falta consolidar matriz fonte/alvo para o artigo e realizar auditoria manual de respostas.
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

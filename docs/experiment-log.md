# Diário de experimentos

## 2026-09-26 — preparação do piloto

**Natureza:** desenvolvimento e validação da infraestrutura. Não é treinamento nem medição de desempenho de LLM.

Foram implementados geração determinística, verificadores de respostas e evidências, splits por problema, avaliação pareada e gráficos. Os testes usam respostas fabricadas somente para verificar o avaliador, em diretórios temporários. Esses valores não constituem resultados científicos.

Falhas encontradas nos testes iniciais:

- O validador aceitava uma lista incompleta de evidências. Foi incluída verificação independente por remoção de fatos; o teste de regressão falhou antes da correção.
- Uma execução sem qualquer resposta era rejeitada, impedindo registrar uma falha total do modelo. Foi corrigida para contar ausências como erros e deixar a taxa condicional indefinida quando necessário.
- O filtro de diretórios da varredura de segredos não reconhecia os separadores do Windows. Foi corrigido e coberto por teste de regressão, mantendo o código e as configurações do projeto na varredura.

Execução local de preparação: seed 42, 100 problemas por tarefa, 300 problemas-base e 1.200 exemplos. O manifesto registra 960 exemplos de treino, 120 de validação e 120 de teste. A partição é `2026-09-27/969659029e986d86` em UTC (a execução ocorreu na noite de 26/09 no horário de São Paulo). O gráfico de composição foi inspecionado visualmente. Nenhuma resposta de LLM foi produzida nessa execução.

## Como registrar cada treinamento futuro

Cada execução deverá registrar:

| Campo | Informação necessária |
|---|---|
| Objetivo | Hipótese e contraste testado |
| Código | Commit efetivamente executado |
| Modelo | Identificador, revisão, licença e precisão |
| Dados | Hash, versão do gerador, splits e tarefas usadas |
| Configuração | Seed, prompts, lote, contexto, otimizador, taxa de aprendizado e adaptadores |
| Execução | Início/fim UTC, hardware, duração, pico de memória e interrupções |
| Resultados | Métricas por tarefa/condição e caminhos dos artefatos |
| Comparação | Baseline equivalente e orçamento utilizado |
| Limitações | Falhas de formato, respostas ausentes, vieses e condições não avaliadas |

Manter curvas de treinamento, dispersão pareada, tabelas e intervalos de confiança ligados aos arquivos que os originaram. Não substituir resultados antigos nem preencher medidas que não foram obtidas. As curvas de treinamento só serão geradas quando houver histórico real de treinamento.

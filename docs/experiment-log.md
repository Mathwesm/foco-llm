# Diário de experimentos

## 2026-09-26 — preparação do piloto

**Natureza:** desenvolvimento e validação da infraestrutura. Não é treinamento nem medição de desempenho de LLM.

Foram implementados geração determinística, verificadores de respostas e evidências, splits por problema, avaliação pareada e gráficos. Os testes usam respostas fabricadas somente para verificar o avaliador, em diretórios temporários. Esses valores não constituem resultados científicos.

Falhas encontradas nos testes iniciais:

- O validador aceitava uma lista incompleta de evidências. Foi incluída verificação independente por remoção de fatos; o teste de regressão falhou antes da correção.
- Uma execução sem qualquer resposta era rejeitada, impedindo registrar uma falha total do modelo. Foi corrigida para contar ausências como erros e deixar a taxa condicional indefinida quando necessário.
- O filtro de diretórios da varredura de segredos não reconhecia os separadores do Windows. Foi corrigido e coberto por teste de regressão, mantendo o código e as configurações do projeto na varredura.

Execução local de preparação: seed 42, 100 problemas por tarefa, 300 problemas-base e 1.200 exemplos. O manifesto registra 960 exemplos de treino, 120 de validação e 120 de teste. A partição é `2026-09-27/969659029e986d86` em UTC (a execução ocorreu na noite de 26/09 no horário de São Paulo). O gráfico de composição foi inspecionado visualmente. Nenhuma resposta de LLM foi produzida nessa execução.

## 2026-09-27 — primeiro baseline local

Execução `5fab79eb7f10fc4d`, código `c7c272c`, Qwen2.5-0.5B-Instruct em FP16 na RTX 3060 de 6 GB. Foram reaproveitados cinco checkpoints da sessão anterior e geradas as 115 respostas restantes. A tentativa inicial de retomada offline falhou com `OfflineModeIsEnabled` durante consulta de metadados pelo tokenizer; nenhuma resposta foi alterada. A retomada com rede concluiu normalmente.

Resultado: 120 respostas brutas, todas encerradas por EOS, sem limite de tokens ou tempo atingido. Todas começaram com bloco Markdown e foram rejeitadas pelo parser estrito. Acurácia estrita zero em todas as condições; não interpretar como incapacidade de raciocínio nem ausência de efeito dos distratores. O teste final não foi usado para inferência.

Geração acumulada: 175,88 segundos e 5.213 tokens de saída. Pico de memória alocada pelo PyTorch: 1.039.616.512 bytes; reservada: 1.080.033.280 bytes. Essas medidas excluem carregamento/download e não representam toda a memória utilizada na GPU. O intervalo de relógio inclui uma pausa solicitada entre sessões; não é usado como duração computacional.

As métricas foram recalculadas a partir das respostas brutas antes da exportação; o gráfico foi inspecionado visualmente. [Relatório, reprodução e artefatos](baseline-2026-09-27.md). Não houve ajuste de pesos nem mudança do parser para melhorar retroativamente esta execução.

## 2026-09-27 — separação de formato, resposta e evidências

As 120 respostas existentes foram reavaliadas sem inferência ou treinamento adicional. O protocolo `content-envelope-v2` mantém o relatório estrito e acrescenta aceitação de um único bloco Markdown completo, sem reparo de conteúdo. Verifica correspondência de prompts e recalcula o baseline estrito antes da publicação.

Foram lidos 120 objetos; 102 tinham identificadores de evidência válidos e 18 usavam referências inexistentes. Exigindo identificadores válidos, houve 22 acertos; avaliando a resposta independentemente desses identificadores, 37/120. Seleção exata de evidências: 35/120. Esses números são leituras diferentes das mesmas saídas, não ganhos do modelo.

A exportação intermediária v1 foi preservada; a v2 acrescenta a métrica de resposta independente. Foram acrescentados testes para cercas incompletas/múltiplas, texto externo, chaves duplicadas, schema incorreto, evidências inválidas, denominadores, integridade de checkpoints e preservação da leitura estrita. O gráfico foi inspecionado. [Protocolo, resultados e limitações](content-evaluation-2026-09-27.md).

## 2026-09-27 — comparação de dois tamanhos em BF16

A tentativa de 1,5B em FP16 (`f0d3988e46770dcb`) concluiu com saídas inválidas repetitivas. O diagnóstico encontrou NaN nos logits; os artefatos foram preservados como falha técnica. Foi adicionada proteção contra scores inválidos e opção explícita de precisão. Ambos os modelos foram então executados integralmente em BF16, com o mesmo código de inferência (`24bc5e0`) e os mesmos controles.

Execuções `d3382a884de2c421` (0,5B) e `6836fbda70e99de1` (1,5B): 120 exemplos cada, todos encerrados por EOS, nenhuma falha numérica. Acerto independente da resposta: 38/120 e 74/120; seleção exata de evidências: 34/120 e 71/120. Todas as saídas usaram blocos Markdown e tiveram rejeição no contrato estrito original. Tempo de geração: 505,14 s e 460,05 s; tokens: 5.129 e 3.904.

O modelo de 1,5B atingiu o limiar exploratório de competência limpa em dedução e acompanhamento; ambos ficaram em 1/10 em aritmética limpa. Em dedução com distratores semelhantes, seis acertos limpos viraram erros no modelo maior. Revisar o benchmark antes do treinamento; não interpretar tamanho como causa isolada nem selecionar configuração pelo teste final.

Exportação, reavaliação de respostas brutas e controles da comparação foram verificados. Foram gerados gráficos por condição e dispersão de acurácia limpa/com distratores. [Relatório e reprodução](model-comparison-2026-09-27.md). Nenhum ajuste de pesos, serviço pago ou edição do artigo foi realizado.

## 2026-09-28 — teste funcional LoRA e retomada

Código `4b602d6`, benchmark-v2 candidato com 1.200 exemplos, modelo Qwen2.5-1.5B-Instruct BF16, LoRA rank 4, seed 42. Foram executadas duas realizações de seis passos: uma interrompida deliberadamente após o terceiro passo e retomada em outro processo; outra contínua como controle. Ambas usam os mesmos seis exemplos de treino e totalizam 170 tokens supervisionados cada.

Os pesos-base permaneceram iguais; os adaptadores mudaram; perdas, gradientes e pesos permaneceram finitos. A comparação encontrou igualdade exata dos tensores dos adaptadores, estado do otimizador e perdas por passo. A recarga de ambos em modelo-base novo teve diferença zero de logits na sonda. Pico alocado: 3.658.595.840 bytes. Perda na sonda de treino: 0,498208 → 0,461185, sem conclusão de eficácia. [Relatório, gráfico e limitações](training-smoke-2026-09-28.md).

Foram encontrados e corrigidos durante desenvolvimento um teste sensível a maiúsculas nos templates e a forma de importar um script em testes com layout src/. Não houve erro de treinamento na GPU. A auditoria manual identificou a concordância “1 marbles”, registrada como limitação da versão candidata. Nenhum teste final foi usado para inferência ou treinamento.

## 2026-09-28 — auditoria v2.1 e baseline sem ajuste

Corrigidos os controles de ordem relativa, quantidade de frases e concordância, preservando a versão anterior. A execução `145ba1dd1550430d`, código `cd2b7be`, avaliou os 120 exemplos de validação v2.1 com o modelo original 1,5B em BF16. Resultado: 6/120 respostas corretas, 9/120 conjuntos de evidências exatos, zero contratos estritos de JSON puro; 112 saídas aceitas pelo avaliador de conteúdo. Todas terminaram por EOS. Nenhuma tarefa atingiu o limiar de competência limpa. Não iniciado treinamento de eficácia; próximo passo recomendado é diagnosticar dificuldade e formato sem abrir o teste final. [Relatório, gráficos e reprodução](baseline-v21-2026-09-28.md).

## 2026-09-28 — ciclo funcional v2.1

Código `646fa00`, treino `aa1bb31bbe1b6c28`, inferência `1388c1f4d0e24a2b`. Seis atualizações LoRA e recarga verificadas; 120 casos de validação avaliados sem alterar o protocolo. Acurácia permaneceu em 6/120 nos mesmos exemplos; evidências exatas passaram de 9 para 10. Perda de sonda de treino caiu de 0,534521 para 0,493718. Sem conclusão de eficácia, sem nuvem e sem teste final. [Relatório e reprodução](adapter-validation-2026-09-28.md); [guia detalhado das tarefas e do histórico](experiment-guide.md).

## 2026-09-28 — diagnóstico de ordem, sem treinamento

Código `f240f71`, execução `bdd7837e740bfc0b`. Trinta problemas limpos da validação v2.1 com frases em ordem inversa e mesmos IDs/gabaritos. Reutilizadas as respostas originais verificadas. Dedução: 3/10 → 2/10, três perdas e dois ganhos; demais tarefas 0/10 → 0/10. Todas as novas respostas terminaram por EOS. [Relatório, pares, limitações e reprodução](order-diagnostic-2026-09-28.md).

## Campos dos próximos treinamentos

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

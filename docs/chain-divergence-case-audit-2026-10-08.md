# Auditoria por caso: divergência de cadeias

Fonte primária: [execução salva do Kaggle, versão 2](https://www.kaggle.com/code/matheusdsousaribeiro/kaggle-chain-divergence-7b/output?scriptVersionId=356460561). O notebook está privado; uma cópia compacta dos arquivos brutos foi preservada em `reports/2026-10-08/chain-divergence-kaggle-output.zip` (SHA-256 `227c20e893aac646367bc97ce3ed9c6c45b2269370479b2f9eb954c78c4a3113`). O conjunto congelado tem SHA-256 `a17296b912c661efb90da2231257ceb9b8facc3cb87ee60de4c2e9a0fc530036`. A análise reproduzível está em `scripts/analyze_chain_divergence.py` e sua saída em `reports/2026-10-08/chain-divergence-case-audit.json`.

## Comparação pareada

Há 40 problemas-base independentes, 20 de dedução e 20 de acompanhamento. Cada base aparece nos quatro braços: distratores sem relação e cadeias concorrentes que divergem cedo, no meio ou tarde. Assim, 160 prompts não equivalem a 160 unidades independentes. Em cada prompt, o Qwen2.5-7B-Instruct base e o mesmo checkpoint com adaptador de dedução receberam o mesmo conjunto de fatos e prompt. Não houve treino novo nesta etapa. Seleção exata exige os quatro fatos corretos e nenhum distrator.

| Tarefa | Base | Adaptador | Ganhos nos mesmos casos | Perdas |
| --- | ---: | ---: | ---: | ---: |
| Dedução, 80 prompts | 27 | 40 | 20 | 7 |
| Acompanhamento, 80 prompts | 4 | 43 | 39 | 0 |
| Total, 160 prompts | 31 | 83 | 59 | 7 |

A diferença agregada é 52/160 = **32,5 pontos percentuais**. Um bootstrap percentil exploratório, com semente 42 e 20.000 reamostragens estratificadas das 20 bases por tarefa, mantendo os quatro braços juntos, produziu intervalo de **23,1 a 41,9 pontos** para essa diferença. Os intervalos por tarefa foram 5,0 a 28,8 pontos em dedução e 33,8 a 63,8 em acompanhamento. Eles quantificam variação dentro deste gerador sintético; não são garantia de desempenho em textos naturais nem teste confirmatório pré-registrado.

## Dois casos concretos

Em `divergence-deduction-0002:late`, os IDs de referência são F1, F4, F6 e F7. O modelo base escolheu F1, F4 e F7, omitindo F6: a relação de S3 de T55431 para T74979. O adaptador escolheu os quatro. O espécime concorrente tinha os mesmos dois primeiros nomes de propriedade, mas seu último salto terminava em T46560, enquanto S3 terminava em T54303. O registro permite afirmar que houve omissão de evidência, não por qual mecanismo interno ela ocorreu.

Em `divergence-tracking-0001:middle`, os IDs de referência são F2, F3, F5 e F8. O modelo base escolheu F5 e F8; o adaptador acrescentou F3, mas ainda omitiu F2, que informa a caixa inicial do token-2. Ambos falharam na seleção exata. O caso mostra que selecionar relações posteriores sem o fato inicial continua sendo um erro observável mesmo após adaptação.

## Limites para o artigo

Em dedução, os acertos exatos caíram entre divergência cedo, meio e tarde: 6→5→4 na base e 12→8→7 no adaptador. O padrão é descritivo com 20 bases. Em acompanhamento, a sequência do adaptador foi 13→9→9 e a base acertou só 2/20 no braço sem relação; esse efeito de piso impede interpretação firme da posição do desvio na base.

O teste anterior com cadeias de cinco relações ficou próximo de zero e utilizou outra redação. Este teste usa três relações e foi desenhado **depois** da análise da falha anterior. Portanto, não se somam as contagens e não se apresenta o resultado posterior como confirmação cega ou prova de generalização robusta. Os IDs produzidos são saídas externas; não medem ativações nem revelam o pensamento interno do modelo.

Para reproduzir a auditoria a partir do arquivo compactado preservado:

```powershell
poetry run python scripts/analyze_chain_divergence.py --archive reports/2026-10-08/chain-divergence-kaggle-output.zip --output reports/<nova-data>/chain-divergence-case-audit.json
```

O script recusa sobrescrever uma saída anterior.

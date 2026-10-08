# Avaliação fora dos moldes: dedução e rastreamento

Este teste exploratório mede se o ganho do especialista em dedução sobrevive a
mudanças de redação e de ordem dos fatos. O código gera 30 bases novas de dedução e
30 de rastreamento; cada base tem uma versão com distrator não relacionado e outra
com cadeia semelhante. São 120 entradas pareadas, mas apenas 60 bases independentes.

O gerador `scripts/run_ood_chain_holdout.py` fixa a semente `20261008`, embaralha a
ordem das frases, usa moldes diferentes dos usados anteriormente e confere os 120
gabaritos com um resolvedor simbólico. O arquivo congelado tem SHA-256
`5a1ee80d4a2af8c27b50fc5e6fbde6eb2369388e01a21c298093447181d64951`.
Ele não foi usado no treinamento ou na escolha do adaptador.

O notebook `notebooks/kaggle_ood_chain_holdout_7b.ipynb` recria o treinamento do
especialista Qwen2.5-7B com os mesmos 80 exemplos e 320 passos do protocolo anterior,
avalia o modelo base e o especialista no mesmo conjunto congelado e grava os
resultados em `/kaggle/working/ood-chain-7b/comparison.json`. Registrar o commit
exato, o hash dos dados e os caminhos dos resultados permite auditar a comparação.

A métrica principal é seleção exata de fatos relevantes em dedução, separada por
tipo de distrator. Também devem ser relatadas validade do formato, omissões e
inclusões indevidas, além do resultado de rastreamento para detectar regressões.
Qualquer diferença deve ser descrita em contagens pareadas por base, sem tratar os
dois distratores de uma base como observações independentes.

Este teste foi desenhado após observar os resultados anteriores. Ele altera a
superfície textual, mas preserva o raciocínio sintético em cadeia de seis fatos.
Por isso, mesmo um ganho positivo não demonstrará um mecanismo interno de
raciocínio nem generalização para textos naturais. Não incorporar o resultado ao
artigo como confirmação independente sem declarar esse caráter pós-hoc.

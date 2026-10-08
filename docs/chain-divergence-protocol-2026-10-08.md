# Distratores com divergência controlada na cadeia

Protocolo exploratório inspirado nas cadeias alternativas de
[Bhuiya et al. (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.147/)
e na variação do salto de divergência de
[Yun et al. (LREC 2026)](https://aclanthology.org/2026.lrec-1.410/).
O desenho, as tarefas e as métricas são próprios do Foco-LLM; não se trata
de replicação daqueles trabalhos.

## Hipótese e desenho

Uma cadeia distratora que copia um prefixo maior da cadeia correta pode
confundir mais a seleção de evidências. Para verificar isso sem trocar o
gabarito, cada problema-base possui quatro braços: quatro frases sem relação
com a pergunta, ou uma cadeia concorrente que diverge no primeiro, segundo
ou terceiro salto. A cadeia correta tem uma frase inicial e três relações.
Há 20 bases de dedução e 20 de rastreamento: 40 unidades pareadas, 160
prompts. Cada prompt contém oito fatos numerados, quatro relevantes e quatro
distratores. O gerador fixa seed, pergunta, resposta, texto e posição dos
quatro fatos corretos entre os braços. Apenas o texto dos quatro distratores
muda. Um resolvedor simbólico independente confirma resposta e IDs de
evidência de todos os 160 casos antes de carregar qualquer modelo.

As relações são explicitamente associadas ao espécime ou token. A cadeia
concorrente usa outro objeto; copia os nomes dos nós até o salto escolhido
e então segue para nós próprios. Essa definição operacional de divergência
mantém um caminho correto único. Também altera a gramática em relação ao
treino do especialista; portanto, o resultado mede generalização para esta
formulação, não um efeito puro de posição em todas as linguagens possíveis.

Exemplo esquemático de dedução, sem os códigos sorteados: `Specimen S1 has
trait T0`; `For specimen S1, trait T0 implies trait T1`; `T1` implica `T2`;
`T2` implica `T3`. Um rival tardio começa em `other-1` e segue `T0 -> T1 ->
T2 -> U3`; um rival precoce segue `T0 -> U1 -> U2 -> U3`. A pergunta pede
o traço final de `S1`; o gabarito é sempre `T3` e os quatro IDs de `S1`.

## Execução e interpretação

O script `scripts/run_chain_divergence.py` congela `dataset.json` e
`manifest.json` com SHA-256. O notebook
`notebooks/kaggle_chain_divergence_7b.ipynb` deve avaliar exatamente o mesmo
conjunto no Qwen2.5-7B-Instruct base e no adaptador de dedução já treinado,
sem novo ajuste. A geração é gulosa com o prompt de seleção da transferência
entre tarefas. Cada resposta bruta é guardada. O placar separa validade do
JSON, seleção exata, falsos positivos e falsos negativos por tarefa e braço;
inclui mudanças pareadas em relação ao controle sem relação.

Se o controle também tiver acerto próximo de zero, o efeito da posição de
divergência fica **inconclusivo**: acerto igual a zero em todos os braços não
mostra robustez. A comparação entre base e especialista é exploratória e
condicionada ao treino prévio. Não há inferência sobre mecanismo interno de
pensamento a partir desses IDs externos.

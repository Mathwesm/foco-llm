# Literatura próxima ao Foco-LLM: método e próximos controles

Revisão direcionada em 8 de outubro de 2026. Esta nota distingue o método
publicado do que foi efetivamente executado no Foco-LLM. Os resultados externos
não são comparáveis numericamente aos nossos: tarefas, modelos, prompts e
denominadores diferem.

| Trabalho | Como foi feito | Relação com o Foco-LLM |
| --- | --- | --- |
| [Shi et al., ICML 2023](https://proceedings.mlr.press/v202/shi23a.html) | Inseriram informação irrelevante em problemas aritméticos e mediram a mudança na resposta. | Justifica pares com resposta preservada; nossos IDs de evidência e executor separam seleção de cálculo. Já citado no artigo. |
| [Yang et al., EMNLP 2025, GSM-DC](https://aclanthology.org/2025.emnlp-main.674/) | Geraram grafos de dependência, mantiveram um caminho correto, injetaram distratores controlados e separaram exatidão do caminho, dos passos e da resposta. Variaram intensidade da distração e profundidade. | É a referência mais direta para nosso gerador sintético, gabarito simbólico e métricas separadas. Ainda não variamos sistematicamente o **número** de distratores mantendo os outros fatores fixos. Já citado no artigo. |
| [Bhuiya et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.147/) | Decompuseram perguntas multi-hop em subperguntas e criaram parágrafos com cadeias alternativas plausíveis que levam a uma resposta errada sem alterar a correta. | Sustenta nosso teste de distrator de cadeia semelhante. O nosso benchmark é sintético e usa fatos numerados; não é reprodução do conjunto de perguntas desse trabalho. |
| [Yun et al., LREC 2026, CRiT-QA](https://aclanthology.org/2026.lrec-1.410/) | Substituíram cadeias factuais por cadeias contrafactuais coerentes e criaram distratores que divergem em diferentes saltos da cadeia. | Sugere um controle útil: variar o ponto de divergência do distrator mantendo pergunta, profundidade e gabarito fixos. Seus escores de QA não devem ser comparados aos nossos de seleção exata. |
| [Chen et al., arXiv 2026](https://arxiv.org/abs/2605.06840) | Rotularam parágrafos de traços de raciocínio de um modelo que os expõe, removeram trechos e mediram mudanças da decisão após continuar a inferência. | Inspira a intervenção em notas do Foco-LLM, mas nosso Qwen2.5-Instruct escreve uma nota **externa** por solicitação. O teste não observa nem altera o raciocínio interno. |
| [Turpin et al., NeurIPS 2023](https://arxiv.org/abs/2305.04388) | Testaram se explicações em cadeia verbalizam fatores que alteram a resposta e mostraram casos de explicação infiel. | Fundamenta a cautela: um ID ou justificativa escrito pelo modelo não prova que o fato causou sua decisão. Já citado no artigo. |

O artigo [Wu et al., arXiv:2508.06361v1](https://arxiv.org/html/2508.06361v1)
enviado anteriormente investiga **decepção** em prompts benignos por meio de
perguntas de contato e métricas de intenção/comportamento. Sua pergunta de
pesquisa e seus rótulos não correspondem à seleção de fatos relevantes; não
devemos importar essas métricas para o Foco-LLM nem citá-lo como validação do
nosso mecanismo.

## O que nossos dados já permitem

- O teste aritmético pareado, a ablação de fatos e a replicação OOD se alinham
  ao princípio de manter o gabarito fixo ao introduzir distração.
- Seleção exata, validade do formato e resposta do executor são observáveis
  diferentes. A pontuação de evidência não equivale à exatidão de cada passo
  de raciocínio de GSM-DC, pois o modelo não produz uma solução passo a passo
  validada simbolicamente.
- A intervenção em notas terá denominador condicionado às notas elegíveis;
  sua taxa de rejeição precisa aparecer ao lado de qualquer mudança pareada.
- Comparações entre 1,5B e 7B continuam confundidas por checkpoint,
  quantização e protocolo; não medem efeito isolado do número de parâmetros.

## Recomendação após os jobs em execução

O próximo experimento com melhor relação entre informação e custo é **variar
o ponto de divergência da cadeia distratora** em dedução e rastreamento, sem
novo treinamento. Congelar as mesmas bases e o mesmo checkpoint; produzir
variantes com divergência no primeiro, no meio e no último salto; verificar
todos os gabaritos com o resolvedor; avaliar modelo base e especialista nos
mesmos itens. Relatar seleção exata, validade, omissão/inclusão e diferenças
pareadas por base. Isso testa uma hipótese mais específica sobre a fragilidade
observada com cadeias semelhantes, inspirada em Bhuiya e Yun, sem alegar
replicação ou análise mecanística de ativações.

Se o 7B também não gerar notas completas, o resultado do protocolo de Chen
permanece **inconclusivo por falha de formato**. Nesse caso, uma versão
simplificada do prompt deve receber novo identificador de protocolo e ser
relatada como teste pós-hoc; não se deve modificar silenciosamente o teste
congelado ou tratá-la como sua continuação.

# Plano experimental

## Objetivo

Investigar se o ajuste de uma LLM para selecionar fatos relevantes melhora a resistência a distratores e se essa melhora transfere para tarefas excluídas do ajuste. Comparar tamanhos de modelos será uma etapa posterior, com controle de precisão, configuração de geração e competência inicial.

Partiremos de modelos pré-treinados. O código fica em inglês; documentação, decisões e relatos em PT-BR. A redação do artigo científico, em inglês, será retomada depois dos experimentos.

## Primeira implementação: infraestrutura do piloto

- Três famílias sintéticas: aritmética, dedução por regras e acompanhamento de objetos.
- Quatro condições por problema: limpa, informação sem relação, quantidade irrelevante e informação parecida com a tarefa.
- Seed explícita e identificador comum para manter todas as variantes no mesmo split.
- Verificação independente da resposta e das evidências por remoção de fatos.
- Exportação de prompts separados dos gabaritos.
- Avaliação de respostas registradas e gráfico de acurácia limpa versus distraída.

**Limite importante:** o gerador atual é para testar o funcionamento do experimento. Usa templates compartilhados entre treino e teste, dificuldade fixa e distratores fáceis de reconhecer por vocabulário. Não é o benchmark científico definitivo. Seus verificadores reconhecem somente a gramática produzida pelo gerador; não validam qualquer texto em linguagem natural.

## Controles do treinamento

Comparar modelo original, instrução explícita sem ajuste, ajuste com dados limpos, ajuste com dados ruidosos e resposta apenas, e ajuste com os mesmos dados ruidosos mais seleção de evidências. Usar fatos corretos previamente selecionados somente como diagnóstico.

Documentar quantidade de exemplos e tokens, etapas de otimização, parâmetros treináveis e custo. A supervisão adicional não deve ser confundida com um orçamento maior. Escolher hiperparâmetros somente na tarefa-fonte; avaliar tarefas-alvo depois de congelar a configuração.

## O que falta antes do primeiro treinamento

1. Confirmar a escolha definitiva após o piloto: Qwen2.5-0.5B-Instruct já foi executado com revisão fixa em GPU.
2. Separar formato e conteúdo na avaliação: o executor com retomada está implementado, mas o primeiro baseline teve todas as respostas rejeitadas por blocos Markdown.
3. Congelar uma nova versão do protocolo antes de repetir a avaliação de desenvolvimento; manter o baseline estrito original e o teste final reservado.
4. Diversificar templates e distratores, separar estruturas e revisar exemplos manualmente.
5. Implementar ajuste, checkpoints, avaliação por sementes e alertas de falha em execuções longas.

## Métricas

Relatar acurácia por tarefa/condição, precisão e revocação das evidências, correspondência exata, respostas ausentes e transições acerto→erro e erro→acerto. A queda de acurácia é expressa em pontos percentuais. A taxa de falha condicionada ao acerto limpo fica indefinida quando não há acertos limpos.

O avaliador conta respostas ausentes como falhas, rejeita identificadores desconhecidos ou duplicados e usa comparação exata normalizada de respostas. “12” é aceito quando o alvo é “12”; “12.0” ou uma explicação completa não são convertidos silenciosamente. O prompt especifica esse contrato.

Intervalos de confiança com reamostragem por problema, comparação entre sementes e análise mecanística ainda não estão implementados. O gráfico de dispersão atual mostra pontos agregados por tarefa/condição, não observações independentes individuais.

## Referências de implementação e método

- [GSM-DC, Yang et al., 2025](https://aclanthology.org/2025.emnlp-main.674/): construção controlada e comparação de treinamento com distratores.
- [NoisyBench / RARE, Lee et al., 2026](https://arxiv.org/abs/2601.07226): precedente direto para resposta e evidências; preprint.
- [Distilling Step-by-Step, Hsieh et al., 2023](https://aclanthology.org/2023.findings-acl.507/): supervisão auxiliar para modelos menores.
- [STaR, Zelikman et al., 2022](https://arxiv.org/abs/2203.14465): geração e filtragem iterativa de exemplos.
- [Let's Verify Step by Step, Lightman et al., 2023](https://arxiv.org/abs/2305.20050): supervisão de etapas para modelos de recompensa.
- [Build a Reasoning Model (From Scratch), Sebastian Raschka](https://sebastianraschka.com/reasoning-from-scratch/): referência prática de avaliação e treinamento a partir de um modelo pré-treinado.

Usar um método publicado não implica contribuição inédita. A originalidade e as conclusões dependerão do protocolo final e dos resultados.

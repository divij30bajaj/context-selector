## Context Selector
This repo contains an algorithm to select relevant context given a sentence from a long document. The algorithm is evaluated on Document Machine Translation using the IWSLT17 English-German dataset.

**Baseline**: For a given sentence, the previous two sentences are selected as the context.
**Experiment 1**: Relevant context is selected using a novel masking technique that masks words in the target sentence and then selects context sentences that predict the masked word most accurately.
**Experiment 2**: Above masking technique is combined with a novel entity overlap score which scores the intersection over union of entity mentions in a potential context sentence and the target sentence.

### Running experiments
To run the baseline:
`python experiments/baseline.py`

To run the experiments, run exp1.py or exp2.py in `experiments` folder.

"""
Doubt: How to predict masked tokens for bigrams and trigrams?
python -m spacy download en_core_web_lg

- Remove full-stop, comma, parenthesis, double quotes, single quotes
- Replace hyphen with space character (Doubt)
- Optimize time (?)

1. If one potential blank is a subset of the other, take the bigger one.
2. If there are two overlapping blanks, choose randomly.
3. With 5% probability, choose an n-gram that does not need context.
4. Compare singular infinitive form of masked word and the Top-5 predictions.
5. Check if any of the Top-5 predictions is not a synonym of the masked word.
6. If a proper noun appears in the passage only once, we don't mask it.
"""
import time

import numpy as np
import torch
from transformers import RobertaForMaskedLM, AutoTokenizer
import spacy

from utils import utils

model_name = "roberta-base"
model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def main(sentences, pos=None):
    spacyTokenizer = spacy.load("en_core_web_lg")
    blanks_dict = {}
    for idx, sentence in enumerate(sentences):
        if pos is not None and idx != pos:
            continue
        sent = spacyTokenizer(sentence)
        temp_potential_blanks, potential_blanks, blanks = [], [], []
        for i, token in enumerate(sent):
            if token.pos_ == 'ADJ' or token.pos_ == 'VERB' or token.pos_ == 'ADV':
                temp_potential_blanks.append(token.text)

        tokenized_input = tokenizer(sentence, return_tensors="pt")
        words = []
        for i, token_id in enumerate(tokenized_input['input_ids'][0]):
            word = tokenizer.decode(token_id.item()).strip()
            if word in temp_potential_blanks:
                words.append(word)
                potential_blanks.append((word, i))
        for i in range(len(potential_blanks)):
            tokens_copy = torch.clone(tokenized_input['input_ids'][0,:])
            tokens_copy[potential_blanks[i][1]] = tokenizer.mask_token_id
            masked_sentence = ""
            for id in tokens_copy[1:-1]:
                if id != tokenizer.mask_token_id:
                    masked_sentence += tokenizer.decode(id)
                else:
                    masked_sentence += " " + tokenizer.decode(id)

            token_input = tokenizer(masked_sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**token_input)
                logits = outputs.logits

            mask_index = torch.where(token_input["input_ids"] == tokenizer.mask_token_id)
            probs = torch.nn.functional.softmax(logits[mask_index[0], mask_index[1], :], dim=-1)
            probs = probs[0, :]
            top_indices = np.argsort(probs.numpy())[-2:][::-1]
            preds = [tokenizer.decode(id) for id in top_indices]

            if utils.check_word(spacyTokenizer, potential_blanks[i][0], preds):
                blanks.append(potential_blanks[i])
        blanks_dict[idx] = blanks

    return blanks_dict


if __name__ == '__main__':
    sentences = [
        "The European Union committed Friday to sending 500 troops to the tumultuous Central African Republic -- a number that the coalition is looking to double, according to its foreign policy chief.",
        "The announcement from the EU foreign policy chief Catherine Ashton came after her address to the U.N. Security Council and after France announced it will send 400 additional troops to its former colony.",
        "Already, France, an EU member, has deployed 1,600 personnel there to support African Union troops following a U.N. Security Council vote in December authorizing military intervention.",
        "In late January, the Security Council voted to not only continue its peacekeeping mission in the war-weary African nation but to authorize the use of force by EU troops there -- setting the stage for Friday's announcement by Ashton.",
        "The Central African Republic plunged into chaos last year after a coalition of rebels dubbed Seleka ousted President Francois Bozize, in the latest in a series of coups since the country gained independence in 1960.",
        "Rebels infiltrated the capital in March, sending Bozize fleeing to Cameroon.",
        "One of the Seleka's leaders, Michel Djotodia, then seized power only to step down as President in January after failing to halt the escalating violence.",
        "Catherine Samba-Panza, the mayor of Bangui, was recently tapped as the country's interim President.",
        "Still, the climate is volatile.",
        "The United Nations recently pointed to an increasing cycle of violence and retaliation and the continuing deterioration of the security situation.",
        "Seleka is a predominantly Muslim coalition, and to counter the attacks on Christian communities, vigilante Christian groups have fought back.",
        "The United Nations -- estimating more than half the country's population is affected by the worsening humanitarian crisis -- has said it fears genocide brewing.",
        "According to the United Nations, more than 700,000 people across the Central African Republic have been displaced -- including about 290,000 alone in the capital of Bangui -- and 2.6 million need immediate humanitarian assistance.",
        "Attempts to purge Muslims from parts of the war-torn country have prompted a Muslim exodus of historic proportions, according to rights group Amnesty International.",
        "And on Friday, a public prosecutor said that a mass grave with at least 13 decomposing bodies had been discovered in Bangui.",
        "Maturin Grenzengue said Central African Republic and allied troops were working to remove concrete slabs of an underground cement tank to determine the exact number of bodies that will be found here.",
        "Valerie Amos, the U.N.'s humanitarian chief, is set to spend three days in the country starting next Tuesday.",
        "The same agency's top official on refugees, Antonio Guterres, said Wednesday the country is a humanitarian catastrophe.",
        "There is an ethnic-religious cleansing taking place.",
        "It must be stopped,\" Guterres said.",
        "There are people who are still being killed here and there -- even some massacres still taking place."
    ]
    start = time.time()
    blanks = main(sentences)
    print(blanks)
    print(time.time() - start)

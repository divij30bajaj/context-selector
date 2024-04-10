from collections import defaultdict

from stanza.server import CoreNLPClient

from model.sent_ranker_ind_non_learn import SentRankerNLBase
import torch
import time
import os

os.environ['PATH'] += ';C:/Program Files/Java/jre-1.8/bin'


class SentRankerNonLearn(SentRankerNLBase):
    def __init__(self, threshold, weight, topK=2):
        self.weight = weight
        self.client = CoreNLPClient(annotators=['coref'], timeout=30000, memory='4G')
        self.entity_sent_map = None
        super().__init__(threshold, topK)

    def extract_coref(self, sentences):
        article = " ".join(sentences).strip()
        annotated = self.client.annotate(article)
        corefs = annotated.corefChain

        sent_entity = defaultdict(list)
        for cluster in corefs:
            clusterId = cluster.chainID
            for mention in cluster.mention:
                sentIndex = mention.sentenceIndex
                sent_entity[sentIndex].append(clusterId)
        return sent_entity

    def score_entity_overlap(self, sent1_id, sent2_id):
        if sent1_id not in self.entity_sent_map.keys() or sent2_id not in self.entity_sent_map.keys():
            score = 0
        else:
            entity1 = set(self.entity_sent_map[sent1_id])
            entity2 = set(self.entity_sent_map[sent2_id])

            common_entities = len(entity1.intersection(entity2))
            total_entities = len(entity1.union(entity2))
            score = common_entities / total_entities
        return score

    def rank_sentences(self, sentences, blanks_dict, pos):
        masked_sentence, masked_ids, reference = self.prepare_masked_input(sentences[pos], blanks_dict[pos])
        print("Sentence: {}".format(sentences[pos]))
        rank = []
        idxs = [0] + list(range(max(1, pos - 5), min(pos + 6, len(sentences))))

        self.entity_sent_map = self.extract_coref(sentences[idxs])
        entity_scores = {}
        for i in idxs:
            if i == pos or i == 0:
                continue
            entity_overlap_score = self.score_entity_overlap(pos, i)
            entity_scores[i] = entity_overlap_score

        result = []
        print("Context:")
        for num_selected in range(self.topK):
            for i in idxs:
                if i < pos:
                    input_sentence = sentences[i] + " </s> " + masked_sentence
                elif i > pos:
                    input_sentence = masked_sentence + " </s> " + sentences[i]
                else:
                    continue

                tokenized_input = self.tokenizer(input_sentence, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**tokenized_input)
                    logits = outputs.logits

                mask_index = torch.where(tokenized_input["input_ids"] == self.tokenizer.mask_token_id)
                probs = torch.nn.functional.softmax(logits[mask_index[0], mask_index[1], :], dim=-1)
                score = []
                for j, mid in enumerate(masked_ids):
                    p = probs[j, mid].item()
                    if reference[mid] == -1:
                        reference[mid] = p
                    else:
                        per_mask_mlm_score = (p - reference[mid]) / reference[mid]
                        score.append(per_mask_mlm_score)

                mlm_score = sum(score) / len(score) if len(score) > 0 else 0
                entity_overlap_score = entity_scores[i] if i in entity_scores.keys() else 0
                total_score = self.weight * entity_overlap_score + (1 - self.weight) * mlm_score
                rank.append((total_score, i))

            rank.sort(reverse=True)
            selected_sent_idx = rank[0][1]
            top_score = rank[0][0]
            if top_score < self.threshold or sentences[selected_sent_idx] == "":
                break
            if selected_sent_idx < pos:
                masked_sentence = sentences[selected_sent_idx] + " </s> " + masked_sentence
            else:
                masked_sentence = masked_sentence + " </s> " + sentences[selected_sent_idx]
            idxs.remove(selected_sent_idx)
            result.append(selected_sent_idx-1)
            print(sentences[selected_sent_idx], top_score)

        print("")
        return result


if __name__ == '__main__':
    sentences = [
        "The European Union committed on Friday to sending 500 troops to the tumultuous Central African Republic -- a number that the coalition is looking to double, according to its foreign policy chief.",
        "The announcement from the EU foreign policy chief Catherine Ashton came after her address to the U.N. Security Council and after France announced it will send 400 additional troops to its former colony.",
        "Already, France, an EU member, has deployed 1,600 personnel there to support African Union troops following a U.N. Security Council vote in December authorizing military intervention.",
        "In late January, the Security Council voted to not only continue its peacekeeping mission in the war-weary African nation but to authorize the use of force by EU troops there -- setting the stage for Friday's announcement by Ashton.",
        "The Central African Republic plunged into chaos last year after a coalition of rebels dubbed Seleka ousted leader Francois Bozize, in the latest in a series of coups since the country gained independence in 1960.",
        "Rebels infiltrated the capital in March, sending Bozize fleeing to Cameroon.",
        "One of the Seleka's leaders, Michel Djotodia, then seized power only to step down as leader in January after failing to halt the escalating violence.",
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
    exp = SentRankerNonLearn(threshold=1.5, weight=0.5)
    start = time.time()
    # exp.rank_for_single_sent(sentences, 4)
    result = exp.rank_for_entire_doc(sentences)
    for i, sent in enumerate(sentences):
        print("Sentence: {}".format(sentences[i]))
        print("Context: {}".format(result[i]))
        print("")
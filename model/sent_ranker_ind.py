import torch
import time
from model.sent_ranker_ind_non_learn import SentRankerNLBase


class SentRankerIndNL(SentRankerNLBase):
    def rank_sentences(self, sentences, blanks_dict, pos):
        masked_sentence, masked_ids, reference = self.prepare_masked_input(sentences[pos],  blanks_dict[pos])
        rank = []
        idxs = [0] + list(range(max(0, pos-5), min(pos+6, len(sentences))))
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
                    print("Mask: {}, Reference: {}".format(mid, p))
                else:
                    print("Mask: {}, Score: {}".format(mid, (p - reference[mid]) / reference[mid]))
                    score.append((p - reference[mid]) / reference[mid])

            avg_score = sum(score)/len(score) if len(score) > 0 else 0
            rank.append((avg_score, i))

        rank.sort(reverse=True)
        result = []
        for (score, i) in rank:
            if score < self.threshold or sentences[i] == "" or len(result) == self.topK:
                break
            print(sentences[i], score)
            result.append(i-1)

        print("")
        return result

if __name__ == '__main__':
    # sentences = [
    #     "The European Union committed on Friday to sending 500 troops to the tumultuous Central African Republic -- a number that the coalition is looking to double, according to its foreign policy chief.",
    #     "The announcement from the EU foreign policy chief Catherine Ashton came after her address to the U.N. Security Council and after France announced it will send 400 additional troops to its former colony.",
    #     "Already, France, an EU member, has deployed 1,600 personnel there to support African Union troops following a U.N. Security Council vote in December authorizing military intervention.",
    #     "In late January, the Security Council voted to not only continue its peacekeeping mission in the war-weary African nation but to authorize the use of force by EU troops there -- setting the stage for Friday's announcement by Ashton.",
    #     "The Central African Republic plunged into chaos last year after a coalition of rebels dubbed Seleka ousted leader Francois Bozize, in the latest in a series of coups since the country gained independence in 1960.",
    #     "Rebels infiltrated the capital in March, sending Bozize fleeing to Cameroon.",
    #     "One of the Seleka's leaders, Michel Djotodia, then seized power only to step down as leader in January after failing to halt the escalating violence.",
    #     "Catherine Samba-Panza, the mayor of Bangui, was recently tapped as the country's interim President.",
    #     "Still, the climate is volatile.",
    #     "The United Nations recently pointed to an increasing cycle of violence and retaliation and the continuing deterioration of the security situation.",
    #     "Seleka is a predominantly Muslim coalition, and to counter the attacks on Christian communities, vigilante Christian groups have fought back.",
    #     "The United Nations -- estimating more than half the country's population is affected by the worsening humanitarian crisis -- has said it fears genocide brewing.",
    #     "According to the United Nations, more than 700,000 people across the Central African Republic have been displaced -- including about 290,000 alone in the capital of Bangui -- and 2.6 million need immediate humanitarian assistance.",
    #     "Attempts to purge Muslims from parts of the war-torn country have prompted a Muslim exodus of historic proportions, according to rights group Amnesty International.",
    #     "And on Friday, a public prosecutor said that a mass grave with at least 13 decomposing bodies had been discovered in Bangui.",
    #     "Maturin Grenzengue said Central African Republic and allied troops were working to remove concrete slabs of an underground cement tank to determine the exact number of bodies that will be found here.",
    #     "Valerie Amos, the U.N.'s humanitarian chief, is set to spend three days in the country starting next Tuesday.",
    #     "The same agency's top official on refugees, Antonio Guterres, said Wednesday the country is a humanitarian catastrophe.",
    #     "There is an ethnic-religious cleansing taking place.",
    #     "It must be stopped,\" Guterres said.",
    #     "There are people who are still being killed here and there -- even some massacres still taking place."
    # ]
    PATH = "../train/train-0.txt"
    f = open(PATH, "r")
    text = f.readlines()[133:163]
    sentences = [t[:-1] for t in text]
    f.close()
    exp = SentRankerIndNL(threshold=1.5)
    start = time.time()
    exp.rank_for_single_sent(sentences, 4)
    # result = exp.rank_for_entire_doc(sentences)
    # print("Time taken: {}".format(time.time()-start))
    # for i, sent in enumerate(sentences):
    #     print("Sentence: {}".format(sentences[i]))
    #     print("Context: {}".format(result[i]))
    #     print("")

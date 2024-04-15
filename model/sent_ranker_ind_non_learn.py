import abc
from transformers import AutoTokenizer, RobertaForMaskedLM

from model import mlm_masking


class SentRankerNLBase:
    def __init__(self, threshold, topK=3):
        self.threshold = threshold
        self.topK = topK
        model_name = "roberta-base"
        self.model = RobertaForMaskedLM.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_masked_input(self, sentence, masks_pos):
        tokenized_input = self.tokenizer(sentence, return_tensors="pt")

        masked_ids = []
        reference = {}
        for _, id in masks_pos:
            mask_id = tokenized_input['input_ids'][0, id].clone()
            masked_ids.append(mask_id)
            tokenized_input['input_ids'][0, id] = self.tokenizer.mask_token_id
            reference[mask_id] = -1

        masked_sentence = ""
        for id in tokenized_input['input_ids'][0, 1:-1]:
            if id != self.tokenizer.mask_token_id:
                masked_sentence += self.tokenizer.decode(id)
            else:
                masked_sentence += " " + self.tokenizer.decode(id)

        return masked_sentence, masked_ids, reference

    @abc.abstractmethod
    def rank_sentences(self, document, blanks_dict, pos):
        raise NotImplementedError

    def rank_for_single_sent(self, document, pos):
        document = [""] + document
        blanks_dict = mlm_masking.main(document, pos)
        return self.rank_sentences(document, blanks_dict, pos)

    def rank_for_entire_doc(self, document):
        document = [""] + document
        result = []
        blanks_dict = mlm_masking.main(document)
        for pos in blanks_dict.keys():
            if pos == 0:
                continue
            context = self.rank_sentences(document, blanks_dict, pos)
            result.append(context)

        return result
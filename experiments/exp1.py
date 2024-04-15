"""
Experiment 1: Selecting top K sentences by masking verbs, adjectives and adverbs and filtering using RoBERTa MLM.
K is dynamic (threshold = ). Each selected sentence is independent of the sentences selected before.
"""
import os
import time

import tqdm
from torch.utils.data import Dataset, DataLoader

from data_builder.doc_nmt import DocNMT
from model.sent_ranker_ind import SentRankerIndNL


def context_builder(document, exp):
    return exp.rank_for_entire_doc(document)


class ExpOneDataset(Dataset):

    def __init__(self, corpus, exp, split):
        self.corpus = corpus[split]
        self.exp = exp
        self.split = split

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        en_document = self.corpus[index]['en']
        de_document = self.corpus[index]['de']
        if os.path.exists("{}-{}.txt".format(self.split, self.split, index)):
            return -1, [[0]], [], []
        result = context_builder(en_document, self.exp)
        return index, result, en_document, de_document


if __name__ == '__main__':
    name = "exp_one"
    exp = SentRankerIndNL(threshold=1.5)
    data_builder = DocNMT(None, dataset="IWSLT17", name=name)
    corp = data_builder.build_raw_documents()
    for split in ["test", "dev", "train"]:
        print("Start Time:", time.time())
        custom_data = ExpOneDataset(corp, exp, split=split)
        dataloader = DataLoader(custom_data, batch_size=1, num_workers=1)

        for batch in tqdm.tqdm(dataloader):
            idx, context, en_doc, de_doc = batch
            idx = idx.numpy()[0]
            if idx != -1:
                f = open("{}-{}.txt".format(split, idx), "w")
                f.write("<context>\n")
                for sent in context:
                    for ctx in sent:
                        f.write(en_doc[ctx][0] + " ")
                    f.write("\n")
                f.write("<source>\n")
                for sent in en_doc:
                    f.write(sent[0]+"\n")
                f.write("<target>\n")
                for sent in de_doc:
                    f.write(sent[0]+"\n")
                f.close()

    print("End Time:", time.time())



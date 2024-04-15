"""
Experiment 2: Selecting sentences iteratively until score goes below threshold or two sentences are selected.
MLM score is normalized by modified-sigmoid and entity overlap score is calculated such that both scores lie between 0 qnd 1
Total score is weighted average of normalized MLM score and entity overlap score
"""
import os
import time

import tqdm
from torch.utils.data import Dataset, DataLoader

from data_builder.doc_nmt import DocNMT
from model.sent_ranker_non_learn import SentRankerNonLearn


def context_builder(document, exp):
    return exp.rank_for_entire_doc(document)


class ExpTwoDataset(Dataset):

    def __init__(self, corpus, exp, split):
        self.corpus = corpus[split]
        self.exp = exp
        self.split = split

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        en_document = self.corpus[index]['en']
        de_document = self.corpus[index]['de']
        if os.path.exists("{}-{}.txt".format(self.split, index)):
            return -1, [[0]], [], []
        result = context_builder(en_document, self.exp)
        return index, result, en_document, de_document


if __name__ == '__main__':
    name = "exp_two"
    exp = SentRankerNonLearn(threshold=0.4, weight=0.5)
    data_builder = DocNMT(None, dataset="IWSLT17", name=name)
    corp = data_builder.build_raw_documents()
    for split in ["test", "dev", "train"]:
        print("Start Time:", time.time())
        custom_data = ExpTwoDataset(corp, exp, split=split)
        dataloader = DataLoader(custom_data, batch_size=1, num_workers=8)

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



from data_builder.data_builder import DataBuilder
import os
import tqdm
from sacremoses import MosesTokenizer, MosesPunctNormalizer


def write_to_files(split, en_doc, de_doc, context):
    fe = open("{}-sent.en".format(split), "w", encoding="utf-8")
    fd = open("{}-sent.de".format(split), "w", encoding="utf-8")
    fc = open("{}-context.en".format(split), "w", encoding="utf-8")
    for sent in context:
        for ctx in sent:
            fc.write(ctx + " ")
        fc.write("\n")
    for sent in en_doc:
        fe.write(sent + "\n")
    for sent in de_doc:
        fd.write(sent + "\n")
    fe.close()
    fc.close()
    fd.close()


class DocNMT(DataBuilder):
    def __init__(self, context_builder, dataset="IWSLT17", name="baseline"):
        self.dataset = dataset
        self.data_path = self.get_data_path()
        self.out_path = "/".join(self.data_path.split("/")[:-1] + [name])
        os.makedirs(self.out_path, exist_ok=True)
        self.context_builder = context_builder
        self.en_tokenizer = MosesTokenizer('en')
        self.de_tokenizer = MosesTokenizer('de')
        self.en_normalizer = MosesPunctNormalizer('en')
        self.de_normalizer = MosesPunctNormalizer('de')
        self.splits = ["train", "dev", "test"]
        super(DocNMT).__init__()

    def get_data_path(self):
        if self.dataset == "IWSLT17":
            return "dataset/IWSLT17/raw"
        else:
            return ValueError("Invalid dataset")

    def build_raw_documents(self):
        splits = {}
        for split in self.splits:
            corpus = []
            en_input_file = os.path.join(self.data_path, "concatenated_en2de_{}_{}.txt".format(split, "en"))
            de_input_file = os.path.join(self.data_path, "concatenated_en2de_{}_{}.txt".format(split, "de"))
            esf = open(en_input_file, "r", encoding="utf-8")
            dsf = open(de_input_file, "r", encoding="utf-8")

            en_sents, de_sents = esf.readlines(), dsf.readlines()
            en_doc, de_doc = [], []
            for en_row, de_row in tqdm.tqdm(zip(en_sents, de_sents)):
                en_row = en_row.strip()
                de_row = de_row.strip()
                if not en_row.startswith("<"):
                    en_doc.append(en_row)
                    de_doc.append(de_row)
                else:
                    if len(en_doc) != 0:
                        corpus.append({"en": en_doc, "de": de_doc})
                        en_doc, de_doc = [], []

            if len(en_doc) != 0:
                corpus.append({"en": en_doc, "de": de_doc})
            esf.close()
            dsf.close()
            splits[split] = corpus
        return splits

    def build_dataset(self):
        for split in self.splits:
            en_input_file = os.path.join(self.data_path, "concatenated_en2de_{}_{}.txt".format(split, "en"))
            de_input_file = os.path.join(self.data_path, "concatenated_en2de_{}_{}.txt".format(split, "de"))
            esf = open(en_input_file, "r", encoding="utf-8")
            dsf = open(de_input_file, "r", encoding="utf-8")
            en_sent_data, de_sent_data, context_data = [], [], []

            en_sents, de_sents = esf.readlines(), dsf.readlines()
            en_doc, de_doc = [], []
            for en_row, de_row in tqdm.tqdm(zip(en_sents, de_sents)):
                en_row = en_row.strip()
                de_row = de_row.strip()
                if not en_row.startswith("<"):
                    en_normalized = self.en_normalizer.normalize(en_row)
                    en_tokenized = self.en_tokenizer.tokenize(en_normalized)
                    de_normalized = self.de_normalizer.normalize(de_row)
                    de_tokenized = self.de_tokenizer.tokenize(de_normalized)

                    en_doc.append(" ".join(en_tokenized))
                    de_doc.append(" ".join(de_tokenized))
                else:
                    if len(en_doc) != 0:
                        context = self.context_builder(en_doc)
                        en_sent_data.extend(en_doc)
                        de_sent_data.extend(de_doc)
                        context_data.extend(context)
                        en_doc, de_doc = [], []

            if len(en_doc) != 0:
                context = self.context_builder(en_doc)
                en_sent_data.extend(en_doc)
                de_sent_data.extend(de_doc)
                context_data.extend(context)
            esf.close()
            dsf.close()

            write_to_files(split, en_sent_data, de_sent_data, context_data)

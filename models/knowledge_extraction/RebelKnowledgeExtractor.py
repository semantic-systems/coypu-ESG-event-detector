from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from models.knowledge_extraction.KnowledgeExtractor import KnowledgeExtractor


class RebelKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, path_to_ckpt: str = "Babelscape/rebel-large"):
        super(RebelKnowledgeExtractor, self).__init__(path_to_ckpt)

    @staticmethod
    def instantiate(path_to_ckpt: str):
        tokenizer = AutoTokenizer.from_pretrained(path_to_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(path_to_ckpt)
        return tokenizer, model
         
    def forward(self, text: str = "We see a covid breakout in Hamburg in 2019.") -> list:
        model_inputs = self.tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors='pt')
        generated_tokens = self.model.generate(
            model_inputs["input_ids"].to(self.model.device),
            attention_mask=model_inputs["attention_mask"].to(self.model.device),
            max_length=256,
            length_penalty=0,
            num_beams=3,
            num_return_sequences=3,
            )

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        triples = list()
        for _, sentence in enumerate(decoded_preds):
            for triple in self.extract_triplets(sentence):
                if triple not in triples:
                    triples.append(triple)
        return triples

    def extract_triplets(self, text: str) -> list:
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'relation': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'relation': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'relation': relation.strip(),'tail': object_.strip()})
        return triplets


if __name__ == "__main__":
    # test
    test = RebelKnowledgeExtractor()
    print(test.forward("Russia destroyed 30% of Ukraine’s power stations within the past week, says Zelensky. Russia is clearly waging this war against civilians. #Russia #russiaisateroriststate #Energy #war"))

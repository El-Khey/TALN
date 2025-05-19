from string import digits
import contractions
import unicodedata
import re


class TextProcessor:
    def __init__(self, dataset, column):
        self.dataset = dataset
        self.column = column

    def transform(self):
        return self.dataset[self.column].apply(self.process_text)

    def process(self, text):
        return self.process_text(text)

    def process_text(self, text):
        processed_text = self.to_lower(text)

        processed_text = self.expand_contractions(processed_text)

        processed_text = self.remove_digits(processed_text)

        processed_text = self.remove_unicode(processed_text)
        processed_text = self.remove_non_ascii_punctuation(processed_text)

        processed_text = self.remove_special_characters(processed_text)
        processed_text = self.remove_extra_spaces(processed_text)

        return processed_text

    def to_lower(self, text):
        return text.lower()

    def remove_digits(self, text):
        num_digits = str.maketrans("", "", digits)
        return text.translate(num_digits)

    def expand_contractions(self, text):
        return contractions.fix(text)

    def remove_unicode(self, text):
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

    def remove_extra_spaces(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def remove_non_ascii_punctuation(self, text):
        processed_text = re.sub(r"[^\x00-\x7F]", "", text)
        return processed_text

    def remove_special_characters(self, text):
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s\'-]", "", text)
        return cleaned_text

    def tokenize(self, text):
        # ici : simple split par défaut
        return text.lower().strip().split()
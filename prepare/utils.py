import re
import json
import nltk
import pandas as pd
from string import punctuation

nltk.download("stopwords")
russian_stopwords = set(nltk.corpus.stopwords.words("russian"))

class Porter:
    """
    Porter stemmer implementation
    """

    PERFECTIVEGROUND = re.compile("((ив|ивши|ившись|ыв|ывши|ывшись)|((?<=[ая])(в|вши|вшись)))$")
    REFLEXIVE = re.compile("(с[яь])$")
    ADJECTIVE = re.compile("(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$")
    PARTICIPLE = re.compile("((ивш|ывш|ующ)|((?<=[ая])(ем|нн|вш|ющ|щ)))$")
    VERB = re.compile(
        "((ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)|((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)))$"
    )
    NOUN = re.compile(
        "(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$"
    )
    RVRE = re.compile("^(.*?[аеиоуыэюя])(.*)$")
    DERIVATIONAL = re.compile(".*[^аеиоуыэюя]+[аеиоуыэюя].*ость?$")
    DER = re.compile("ость?$")
    SUPERLATIVE = re.compile("(ейше|ейш)$")
    I = re.compile("и$")
    P = re.compile("ь$")
    NN = re.compile("нн$")

    @staticmethod
    def stem(word):
        word = word.replace("ё", "е")
        m = re.match(Porter.RVRE, word)
        if m and m.groups():
            pre = m.group(1)
            rv = m.group(2)
            temp = Porter.PERFECTIVEGROUND.sub("", rv, 1)
            if temp == rv:
                rv = Porter.REFLEXIVE.sub("", rv, 1)
                temp = Porter.ADJECTIVE.sub("", rv, 1)
                if temp != rv:
                    rv = temp
                    rv = Porter.PARTICIPLE.sub("", rv, 1)
                else:
                    temp = Porter.VERB.sub("", rv, 1)
                    if temp == rv:
                        rv = Porter.NOUN.sub("", rv, 1)
                    else:
                        rv = temp
            else:
                rv = temp

            rv = Porter.I.sub("", rv, 1)
            if re.match(Porter.DERIVATIONAL, rv):
                rv = Porter.DER.sub("", rv, 1)

            temp = Porter.P.sub("", rv, 1)
            if temp == rv:
                rv = Porter.SUPERLATIVE.sub("", rv, 1)
                rv = Porter.NN.sub("н", rv, 1)
            else:
                rv = temp
            word = pre + rv
        return word

def stem(text):
    tokens = [Porter.stem(w) for w in text.split()]
    tokens = [token for token in tokens if token not in russian_stopwords\
        and token.strip() not in punctuation]
    return " ".join(tokens)

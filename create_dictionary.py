import re

import requests
from bs4 import BeautifulSoup


def get_urls():
    urls = []

    url = "https://en.wiktionary.org/wiki/Category:German_terms_with_IPA_pronunciation"

    while url != None:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        entries = soup.find("div", {"class": "mw-category"})
        for entry in entries.find_all("a"):
            if not (re.match("[0-9]", entry.getText()) or " " in entry.getText()):
                link = entry["href"]
                urls.append("https://en.wiktionary.org" + link)
                print(link)

        url = None
        for a in soup.find_all("a"):
            if "next page" in a.getText():
                link = a["href"]
                url = "https://en.wiktionary.org" + link

    return urls


def get_word(soup):
    return soup.find("h1").getText()


def get_pron(soup):

    german = soup.find("span", {"id": "German"})
    if german is not None:
        ipa = german.find_next("span", {"class": "IPA"})
        if ipa is not None:
            ipa = ipa.getText()
            pron = []
            i = 0
            while i < len(ipa):
                if i + 2 < len(ipa) and re.match("p͡f|t͡s|t͡ʃ", ipa[i:i+3]):
                    pron.append(ipa2pron[ipa[i:i+3]])
                    i += 3
                    continue
                if i + 1 < len(ipa) and re.match("aː|ɑː|eː|ɛː|iː|oː|øː|uː|yː|aɪ|aʊ|ɔʏ|ɔɪ|dʒ|pf|ts|tʃ", ipa[i:i+2]):
                    pron.append(ipa2pron[ipa[i:i+2]])
                    i += 2
                    continue
                if ipa[i] in ipa2pron:
                    pron.append(ipa2pron[ipa[i]])
                i += 1

            if len(pron) > 1 and pron[0] == "?":
                pron = pron[1:]
            return " ".join(pron)

    return None


ipa2pron = {"a": "A", "ɑ": "A", "aː": "AA", "ɑː": "AA",
            "e": "E", "eː": "EE", "ɛ": "EH", "ɛː": "AEE",
            "i": "I", "iː": "II", "ɪ": "IH",
            "o": "O", "oː": "OO", "ɔ": "OH",
            "ø": "OE", "øː": "OEE", "œ": "OEH",
            "u": "U", "uː": "UU", "ʊ": "UH",
            "y": "UE", "yː": "UEE", "ʏ": "UEH",
            "aɪ": "AI", "aʊ": "AU", "ɔʏ": "OI", "ɔɪ": "OI",
            "ɐ": "AX", "ə": "EX",
            "b": "B", "ç": "C", "d": "D", "dʒ": "JH",
            "f": "F", "ɡ": "G", "h": "H", "j": "J",
            "k": "K", "l": "L", "m": "M", "n": "N",
            "ŋ": "NG", "p": "P", "pf": "PF", "p͡f": "PF",
            "ʁ": "R", "r": "R", "ʀ": "R",
            "s": "S", "ʃ": "SH", "t": "T",
            "ts": "TS", "t͡s": "TS", "tʃ": "CH", "t͡ʃ": "CH",
            "v": "V", "x": "X", "χ": "X",
            "z": "Z", "ʒ": "ZH", "ʔ": "?"}

urls = get_urls()

file = open("pronunciation_dictionary.txt", "w", encoding="utf-8")

for url in urls:
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    word = get_word(soup)
    pron = get_pron(soup)
    if word is not None and pron is not None:
        line = get_word(soup) + " " + get_pron(soup)
        print(line)
        file.write(line + "\n")

file.close()

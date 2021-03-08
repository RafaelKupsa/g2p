from g2p_converter import make_dict
import re

vowels_regex = r"A|AA|E|EE|EH|AEE|I|II|IH|O|OO|OH|OE|OEE|OEH|U|UU|UH|UE|UEE|UEH|AI|AU|OI|AX|EX"
vowels_without_schwa_regex = r"A|AA|E|EE|EH|AEE|I|II|IH|O|OO|OH|OE|OEE|OEH|U|UU|UH|UE|UEE|UEH|AI|AU|OI|AX"
consonants_regex = r"B|C|D|JH|F|G|H|J|K|L|M|N|NG|P|PF|R|S|SH|T|TS|CH|V|X|Z|ZH|\?"


def r_clean_up(string):
    string = re.sub(r"EX R$", r"AX", string)

    v_R_c_regex = r"((?:^| )(?:{v})) R ((?:{c})(?:$| ))".format(v=vowels_without_schwa_regex, c=consonants_regex)
    v_R_end_regex = r"((?:^| )(?:{v})) R$".format(v=vowels_without_schwa_regex)
    
    string = re.sub(v_R_c_regex, r"\1 AX \2", string)
    string = re.sub(v_R_end_regex, r"\1 AX", string)

    return string


def schwa_clean_up(string):
    c_M_c_regex = r"((?:^| )(?:{c})) M ((?:{c})(?:$| ))".format(c=consonants_regex)
    c_NG_c_regex = r"((?:^| )(?:{c})) NG ((?:{c})(?:$| ))".format(c=consonants_regex)
    c_N_c_regex = r"((?:^| )(?:{c})) N ((?:{c})(?:$| ))".format(c=consonants_regex)
    c_L_c_regex = r"((?:^| )(?:{c})) L ((?:{c})(?:$| ))".format(c=consonants_regex)
    c_M_end_regex = r"((?:^| )(?:{c})) M$".format(c=consonants_regex)
    c_NG_end_regex = r"((?:^| )(?:{c})) NG$".format(c=consonants_regex)
    c_N_end_regex = r"((?:^| )(?:{c})) N$".format(c=consonants_regex)
    c_L_end_regex = r"((?:^| )(?:{c})) L$".format(c=consonants_regex)

    string = re.sub(c_M_c_regex, r"\1 EX N \2", string)
    string = re.sub(c_NG_c_regex, r"\1 EX N \2", string)
    string = re.sub(c_N_c_regex, r"\1 EX N \2", string)
    string = re.sub(c_L_c_regex, r"\1 EX L \2", string)

    string = re.sub(c_M_end_regex, r"\1 EX N", string)
    string = re.sub(c_NG_end_regex, r"\1 EX N", string)
    string = re.sub(c_N_end_regex, r"\1 EX N", string)
    string = re.sub(c_L_end_regex, r"\1 EX L", string)

    return string


word2pron = list(make_dict().items())

clean_words = []
clean_prons = []
for word, pron in word2pron:
    if not re.match("[a-zäöüß]+", word.lower()):
        continue
    clean_word = re.sub("[^a-zäöüß]", "_", word.lower())
    if clean_word in clean_words:
        continue

    pron_string = " ".join(pron)
    pron_string = r_clean_up(pron_string)
    pron_string = schwa_clean_up(pron_string)
    clean_pron = pron_string.split()

    if len(clean_pron) < 1:
        continue

    clean_words.append(clean_word)
    clean_prons.append(clean_pron)

    print(word, ",", clean_word, "-->", pron, ",", clean_pron)


file = open("pronunciation_dictionary_clean.txt", "w", encoding="utf-8")

for word, pron in zip(clean_words, clean_prons):
    file.write(word + " " + " ".join(pron) + "\n")

file.close()





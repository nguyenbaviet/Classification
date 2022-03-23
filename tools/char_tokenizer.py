import torch

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"
TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = [
    "aˇ",
    "aˆ",
    "Aˇ",
    "Aˆ",
    "eˆ",
    "Eˆ",
    "oˆ",
    "o˒",
    "Oˆ",
    "O˒",
    "u˒",
    "U˒",
    "D^",
    "d^",
]
CTLABELS = [
    # " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
    "ˋ",
    "ˊ",
    "﹒",
    "ˀ",
    "˜",
    "ˇ",
    "ˆ",
    "˒",
    "‑",
]


def make_groups():
    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()


def parse_tone(word):
    res = ""
    tone = ""
    for char in word:
        if char in dictionary:
            for group in groups:
                if char in group:
                    if tone == "":
                        tone = TONES[group.index(char)]
                    res += group[0]
        else:
            res += char
    res += tone
    return res


def full_parse(word):
    word = parse_tone(word)
    res = ""
    for char in word:
        if char in SOURCES:
            res += TARGETS[SOURCES.index(char)]
        else:
            res += char
    return res


def correct_tone_position(word):
    word = word[:-1]
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def viqr_decode(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    replace_char = correct_tone_position(recognition)
    if recognition[-1] in TONES:
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(
                    replace_char, group[TONES.index(tone)]
                )
    return recognition


class TokenLabelConverter(object):
    """
    Convert between text-label and text-index
    """

    def __init__(self):
        self.GO = "[GO]"
        self.SPACE = "[s]"

        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + CTLABELS

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = 27

    def encode(self, text):
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(
            self.dict[self.GO]
        )
        for i, t in enumerate(text):
            txt = (
                [self.GO] + list(full_parse(t).replace(" ", self.SPACE)) + [self.SPACE]
            )
            txt = [self.dict[char] for char in txt]
            batch_text[i][: len(txt)] = torch.LongTensor(txt)
        return batch_text

    def decode(self, text_index):
        texts = []
        for i in range(len(text_index)):
            text = []
            for ti in text_index[i]:
                char = self.character[ti]
                text.append(char)
            text = "".join(text)
            texts.append(text)
        return texts


if __name__ == "__main__":
    tlcvt = TokenLabelConverter()
    text_index = tlcvt.encode(["Rồng", "Phượng"])
    text = tlcvt.decode(text_index)
    print(text)

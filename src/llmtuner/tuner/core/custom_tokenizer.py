
import json
from pathlib import Path
import os
from typing import Dict, List, Sequence, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


def get_uci_labels():
    """ Returns a list of possible moves encoded as UCI (including
    promotions).
    Source:
        https://github.com/Zeta36/chess-alpha-zero/blob/
        master/src/chess_zero/config.py#L88
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                [(l1, t) for t in range(8)] + \
                [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                [(l1 + a, n1 + b) for (a, b) in
                    [(-2, -1), (-1, -2), (-2, 1), (1, -2),
                     (2, -1), (-1, 2), (2, 1), (1, 2)]]

            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):  # noqa: E501
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]  # noqa: E501
                    labels_array.append(move)

    for l1 in range(8):
        letter = letters[l1]
        for p in promoted_to:
            labels_array.append(letter + '2' + letter + '1' + p)
            labels_array.append(letter + '7' + letter + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(letter + '2' + l_l + '1' + p)
                labels_array.append(letter + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(letter + '2' + l_r + '1' + p)
                labels_array.append(letter + '7' + l_r + '8' + p)
    return labels_array

def get_chess_vocab():
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    b_pieces = ['k', 'q', 'r', 'b', 'n', 'p']
    w_pieces = ['K', 'Q', 'R', 'B', 'N', 'P']
    special = ['-', '.', 'w', 'b', ' ']
    actions = get_uci_labels()
    vocabs = numbers + b_pieces + w_pieces + special + actions
    return vocabs


class CustomTokenizer(PreTrainedTokenizer):
    model_input_names: List[str] = ["input_ids", "attention_mask"]

    def __init__(self, vocab: Sequence[str], model_max_length: int, **kwargs):
        """
        Args:
            vocab (Sequence[str]): List of desired tokens. Following are list of all of the special tokens with
                their corresponding ids:
                    "[PAD]": 0,
                    "[SEP]": 1,
                    "[MASK]": 2,
                    "[EOS]": 3,
                    "[UNK]": 4,
                an id (starting at 5) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        self.vocab = vocab
        self.model_max_length = model_max_length
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[PAD]": 0,
            "[SEP]": 1,
            "[MASK]": 2,
            "[EOS]": 3,
            "[UNK]": 4,
            **{ch: i + 5 for i, ch in enumerate(vocab)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            eos_token=eos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        # suppose text is a like "split1 split2 split3", convert to character if split* not in vocab
        splits = text.split(' ')
        tokens = []
        for split in splits:
            if split is not '':
                if split in self._vocab_str_to_int:
                    tokens.extend([split, ' '])
                else:
                    tokens.extend(list(split) + [' '])
        return tokens[:-1]

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_config(self) -> Dict:
        return {
            "vocab": self.vocab,
            "model_max_length": self.model_max_length,
        }
    
    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int
    
    @classmethod
    def from_config(cls, config: Dict) -> "CustomTokenizer":
        cfg = {}
        cfg["vocab"] = config['vocab']
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)
    

if __name__ == '__main__':
    tokenizer = CustomTokenizer(get_chess_vocab(), model_max_length=128)
    print(tokenizer.vocab_size)
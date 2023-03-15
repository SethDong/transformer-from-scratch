import unittest
from typing import Dict, List, Tuple, Optional

import torch

from vocabulary import Vocabulary

# 得到一个 seq_len * seq_len 大小的矩阵， 该矩阵的主对角线和之下的部分 全部为True 其余部分为False
# 这样就起到 遮蔽作用
def construct_future_mask(seq_len: int):
    """
    Construct a binary mask that contains 1's for all valid connections and 0's for all outgoing future connections.
    This mask will be applied to the attention logits in decoder self-attention such that all logits with a 0 mask
    are set to -inf.

    在自注意力机制中，为了避免模型在预测时使用后面的信息，通常需要将当前时间步之后的部分屏蔽掉。这个过程就需要使用到一个二元掩码（binary mask），
    其中所有当前时间步之后的位置都被掩盖掉了。

    在该函数中，构建的二元掩码是一个上三角矩阵，其中主对角线及其以下部分被赋值为1，而主对角线以上的部分都被赋值为0。
    这是因为，主对角线及其以下部分对应着所有前面的位置，而主对角线以上的部分对应着当前时间步之后的位置。

    然后，将这个掩码应用到解码器的自注意力机制中，用于掩盖掉那些当前时间步之后的位置。
    具体来说，对于那些被掩盖掉的位置，会将其对应的注意力得分（即注意力logits）设置为负无穷大（-inf），从而在softmax归一化时将它们归为0，不对后续计算产生影响。
    这样就能保证模型只关注当前时间步之前的信息，避免了未来信息对预测的影响。

    :param seq_len: length of the input sequence
    :return: (seq_len,seq_len) mask
    """

    ## torch.triu 用来返回对输入矩阵取上三角， diagonal参数为对角线偏移量=1，表示取第一条对角线之上的部分（去掉主对角线及其以下部分）
    ### torch.full 创建指定大小的张量，并填充一个指定的标量值 此处创建了一个 seq_len * seq_len 所有值都为1 的矩阵
    subsequent_mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
    return subsequent_mask == 0


def construct_batches(
    corpus: List[Dict[str, str]],  # 对齐的 源 和 目的 语句打包而成的字典的列表
    vocab: Vocabulary,
    batch_size: int,
    src_lang_key: str,              # 源语言的标识
    tgt_lang_key: str,              # 目的语言的标识
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:  # 返回含有两个字典的元组，第一个字典表示batches，第二个字典表示注意力遮罩
    """
    Constructs batches given a corpus.

    :param corpus: The input corpus is a list of aligned source and target sequences, packed in a dictionary.
    :param vocab: The vocabulary object.
    :param batch_size: The number of sequences in a batch
    :param src_lang_key: The source language key is a string that the source sequences are keyed under. E.g. "en"
    :param tgt_lang_key: The target language key is a string that the target sequences are keyed under. E.g. "nl"
    :param device: whether or not to move tensors to gpu
    :return: A tuple containing two dictionaries. The first represents the batches, the second the attention masks.
    """
    pad_token_id = vocab.token2index[vocab.PAD]
    batches: Dict[str, List] = {"src": [], "tgt": []}
    masks: Dict[str, List] = {"src": [], "tgt": []}
    for i in range(0, len(corpus), batch_size):  # （步长为batch_size） 就是说每隔batch_size取一个
        src_batch = torch.IntTensor(    # 通过 torch.IntTensor可以将数据转换为整型tensor，以适应torch运算
            # 通过 vocabulary 对象的 batch_encode() 方法将 每个batch的 src 语句 进行padding和id化
            vocab.batch_encode(
                [pair[src_lang_key] for pair in corpus[i : i + batch_size]],
                add_special_tokens=True,
                padding=True,
            )
        )
        tgt_batch = torch.IntTensor(
            vocab.batch_encode(
                [pair[tgt_lang_key] for pair in corpus[i : i + batch_size]],
                add_special_tokens=True,
                padding=True,
            )
        )

        src_padding_mask = src_batch != pad_token_id    # 判断整数序列 src_batch 中是否存在填充符 pad_token_id, 返回的是src_batch形状的，boolean数组
        future_mask = construct_future_mask(tgt_batch.shape[-1])  # 获取到seq_len, 传入遮罩函数 得到遮罩矩阵

        # Move tensors to gpu; if available
        if device is not None:
            src_batch = src_batch.to(device)  # type: ignore
            tgt_batch = tgt_batch.to(device)  # type: ignore
            src_padding_mask = src_padding_mask.to(device)
            future_mask = future_mask.to(device)
        # 将每个batch的数据都加入到字典列表中
        batches["src"].append(src_batch)
        batches["tgt"].append(tgt_batch)
        masks["src"].append(src_padding_mask)
        masks["tgt"].append(future_mask)
    return batches, masks


class TestUtils(unittest.TestCase):
    def test_construct_future_mask(self):
        mask = construct_future_mask(3)
        torch.testing.assert_close(
            mask,
            torch.BoolTensor(
                [[True, False, False], [True, True, False], [True, True, True]]
            ),
        )

    def test_construct_future_mask_first_decoding_step(self):
        mask = construct_future_mask(1)
        torch.testing.assert_close(
            mask, torch.BoolTensor([[True]]),
        )

    def test_construct_batches(self):
        corpus = [
            {"en": "This is an english sentence.", "nl": "Dit is een Nederlandse zin."},
            {"en": "The weather is nice today.", "nl": "Het is lekker weer vandaag."},
            {
                "en": "Yesterday I drove to a city called Amsterdam in my brand new car.",
                "nl": "Ik reed gisteren in mijn gloednieuwe auto naar Amsterdam.",
            },
            {
                "en": "You can pick up your laptop at noon tomorrow.",
                "nl": "Je kunt je laptop morgenmiddag komen ophalen.",
            },
        ]
        en_sentences, nl_sentences = (
            [d["en"] for d in corpus],
            [d["nl"] for d in corpus],
        )
        vocab = Vocabulary(en_sentences + nl_sentences)
        batches, masks = construct_batches(
            corpus, vocab, batch_size=2, src_lang_key="en", tgt_lang_key="nl"
        )
        torch.testing.assert_close(
            batches["src"],
            [
                torch.IntTensor(
                    [[0, 3, 4, 5, 6, 7, 8, 1], [0, 9, 10, 4, 11, 12, 8, 1]]
                ),
                torch.IntTensor(
                    [
                        [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 8, 1],
                        [0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 8, 1, 2, 2, 2, 2],
                    ]
                ),
            ],
        )

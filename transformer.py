import random
import unittest
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from vocabulary import Vocabulary
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from utils import construct_future_mask


# 继承自 torch.nn.Module
class Transformer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            ff_dim: int,
            num_heads: int,
            num_layers: int,
            max_decoding_length: int,
            vocab_size: int,
            padding_idx: int,
            bos_idx: int,
            dropout_p: float,
            tie_output_to_embedding: Optional[bool] = None,
    ):
        super().__init__()
        # Because the encoder embedding, and decoder embedding and decoder pre-softmax transformation share embeddings
        # weights, initialize one here and pass it on.
        """
            nn.Embedding是PyTorch中的一个类，用于定义一个可训练的嵌入层，将输入的离散化整数序列转换为密集的向量表示。
            该类的构造函数接收两个参数：num_embeddings和embedding_dim，分别表示词汇表的大小和每个单词的嵌入维度。
        嵌入层的输入是一个形状为(batch_size, seq_len)的LongTensor类型张量，其中每个元素是词汇表中某个单词的索引，
        输出是一个形状为(batch_size, seq_len, embedding_dim)的FloatTensor类型张量，其中每个元素是对应单词的嵌入向量。
            嵌入向量是通过随机初始化的方式得到的，训练过程中根据损失函数的反向传播调整向量的值，使得单词之间的相似性能够在向量空间中得到体现，
        从而提高模型的性能。在自然语言处理任务中，嵌入层通常作为神经网络的第一层，用于将单词序列转换为向量序列，
        然后输入到后续的神经网络模型中进行进一步的处理。
        """
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)

        self.encoder = TransformerEncoder(
            self.embed, hidden_dim, ff_dim, num_heads, num_layers, dropout_p
        )
        self.decoder = TransformerDecoder(
            self.embed,
            hidden_dim,
            ff_dim,
            num_heads,
            num_layers,
            vocab_size,
            dropout_p,
            tie_output_to_embedding,
        )

        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.max_decoding_length = max_decoding_length
        self.hidden_dim = hidden_dim
        self._reset_parameters()  # 对维度大于1的参数进行均匀初始化

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # Glorot初始化 或Xavier初始化
                # 它通过使用均匀分布来填充输入张量，使得输出张量的值从均匀分布
                # 这种初始化方法可以帮助神经网络更快地收敛，并提高性能。在实际应用中，这种方法被广泛使用
                xavier_uniform_(p)


class TestTransformer(unittest.TestCase):
    def test_transformer_inference(self):
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create (shared) vocabulary and special token indices given a dummy corpus
        corpus = [
            "Hello my name is Joris and I was born with the name Joris.",
            "Dit is een Nederlandse zin.",
        ]
        en_vocab = Vocabulary(corpus)
        en_vocab_size = len(en_vocab.token2index.items())
        with torch.no_grad():
            transformer = Transformer(
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                max_decoding_length=10,
                vocab_size=en_vocab_size,
                padding_idx=en_vocab.token2index[en_vocab.PAD],
                bos_idx=en_vocab.token2index[en_vocab.BOS],
                dropout_p=0.1,
                tie_output_to_embedding=True,
            )
            transformer.eval()

            # Prepare encoder input, mask and generate output hidden states
            encoder_input = torch.IntTensor(
                en_vocab.batch_encode(corpus, add_special_tokens=False)
            )
            src_padding_mask = encoder_input != transformer.padding_idx
            encoder_output = transformer.encoder.forward(
                encoder_input, src_padding_mask=src_padding_mask
            )
            self.assertEqual(torch.any(torch.isnan(encoder_output)), False)

            # Prepare decoder input and mask and start decoding
            decoder_input = torch.IntTensor(
                [[transformer.bos_idx], [transformer.bos_idx]]
            )
            future_mask = construct_future_mask(seq_len=1)
            for i in range(transformer.max_decoding_length):
                decoder_output = transformer.decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                # Take the argmax over the softmax of the last token to obtain the next-token prediction
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)

                # Append the prediction to the already decoded tokens and construct the new mask
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])
        en_vocab.index2token.get(11)
        print([en_vocab.index2token.get(i) for i in decoder_input[0]])
        self.assertEqual(decoder_input.shape, (2, transformer.max_decoding_length + 1))
        # see test_one_layer_transformer_decoder_inference in decoder.py for more information. with num_layers=1 this
        # will be true.
        self.assertEqual(torch.all(decoder_input == transformer.bos_idx), False)




if __name__ == "__main__":
    unittest.main()

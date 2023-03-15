import unittest
from typing import List, Dict, Any
import random
from random import choices

import numpy as np
import torch
from torch import nn

from lr_scheduler import NoamOpt
from transformer import Transformer
from vocabulary import Vocabulary
from utils import construct_batches


def train(
    transformer: nn.Module,
    scheduler: Any,
    criterion: Any,
    batches: Dict[str, List[torch.Tensor]],
    masks: Dict[str, List[torch.Tensor]],
    n_epochs: int,
):
    """
    Main training loop

    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param n_epochs: the number of epochs to train the model for
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(True)    # 将模型模式调整为训练模式（以便可以进行反向传播和参数学习）
    num_iters = 0

    for e in range(n_epochs):
        for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
            zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
        ):
            encoder_output = transformer.encoder(src_batch, src_padding_mask=src_mask)  # type: ignore

            # Perform one decoder forward pass to obtain *all* next-token predictions for every index i given its
            # previous *gold standard* tokens [1,..., i] (i.e. teacher forcing) in parallel/at once.
            decoder_output = transformer.decoder(
                tgt_batch,
                encoder_output,
                src_padding_mask=src_mask,
                future_mask=tgt_mask,
            )  # type: ignore

            # Align labels with predictions: the last decoder prediction is meaningless because we have no target token
            # for it. The BOS token in the target is also not something we want to compute a loss for.
            decoder_output = decoder_output[:, :-1, :]
            tgt_batch = tgt_batch[:, 1:]

            # Set pad tokens in the target to -100 so they don't incur a loss
            # tgt_batch[tgt_batch == transformer.padding_idx] = -100

            # Compute the average cross-entropy loss over all next-token predictions at each index i given [1, ..., i]
            # for the entire batch. Note that the original paper uses label smoothing (I was too lazy).
            batch_loss = criterion(
                decoder_output.contiguous().permute(0, 2, 1),
                tgt_batch.contiguous().long(),
            )

            # Rough estimate of per-token accuracy in the current training batch
            batch_accuracy = (
                torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)
            ) / torch.numel(tgt_batch)

            if num_iters % 100 == 0:
                print(
                    f"epoch: {e}, num_iters: {num_iters}, batch_loss: {batch_loss}, batch_accuracy: {batch_accuracy}"
                )

            # Update parameters
            batch_loss.backward()
            scheduler.step()
            scheduler.optimizer.zero_grad()
            num_iters += 1
    return batch_loss, batch_accuracy


class TestTransformerTraining(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def test_copy_task(self):
        """
        Test training by trying to (over)fit a simple copy dataset - bringing the loss to ~zero. (GPU required)
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if device.type == "cpu":
            print("This unit test was not run because it requires a GPU")
            return

        # Hyperparameters
        synthetic_corpus_size = 600
        batch_size = 200
        n_epochs = 200
        n_tokens_in_batch = 10

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        # Note: the original paper uses byte pair encodings, we simply take each word to be a token.
        # 原文中使用了BPE（字节对编码), 此处做了简化，简单使用了每个词作为token
        corpus = ["These are the tokens that will end up in our vocabulary"]
        vocab = Vocabulary(corpus)   # 语料库构建词典（就这上面一句话哈哈哈）
        vocab_size = len(
            list(vocab.token2index.keys())
        )  # 14 tokens including bos, eos and pad
        valid_tokens = list(vocab.token2index.keys())[3:]   # 除去特殊token之外的有效tokens
        corpus += [
            " ".join(choices(valid_tokens, k=n_tokens_in_batch))  # 随机选取 n_tokens_in_batch 个有效tokens， 用空格连接成为一个字符串
            for _ in range(synthetic_corpus_size)
        ] # 构成了一个 大小为synthetic_corpus_size+1的语料库，除了第一句其他都是随机挑选的，可能并不能称之为句子

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        corpus = [{"src": sent, "tgt": sent} for sent in corpus]
        batches, masks = construct_batches(   # 返回所有batch的数据和mask
            corpus,  # 语料库
            vocab,   # 词典对象
            batch_size=batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )

        # Initialize transformer 并且加入到cuda
        transformer = Transformer(
            hidden_dim=512,  # 隐层维度 512
            ff_dim=2048,     # 全连接维度 2048
            num_heads=8,        # 头数 8
            num_layers=2,       # 层数 2
            max_decoding_length=25,  # 最大解码长度
            vocab_size=vocab_size,  # 词表大小
            padding_idx=vocab.token2index[vocab.PAD],
            bos_idx=vocab.token2index[vocab.BOS],
            dropout_p=0.1,  # dropout值
            tie_output_to_embedding=True,
        ).to(device)

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        # 原文使用的 标签平滑 ———— 在分类时不使用 one-hot 而是使用连续数值分布向量 来避免过度自信 overconfident
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        """
            Adam（Adaptive Moment Estimation）是一种自适应优化算法，结合了梯度的一阶矩估计和二阶矩估计，并且可以自适应地调整每个参数的学习率。
            Adam优化器通过使用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率，这使得它能够处理稀疏梯度、非平稳目标函数以及具有不同尺度的参数的问题。
            在使用 torch.optim.Adam 时，需要传入待优化的模型参数和学习率等超参数。常见的参数包括：
                        params：待优化的模型参数，通常使用model.parameters()返回的迭代器来获取。
                        lr：学习率，表示每次参数更新的步长大小。
                        betas：用于计算一阶和二阶矩估计的指数衰减率。
                        eps：为了数值稳定性而添加到分母中的小值。
                        weight_decay：L2正则化系数。
                        amsgrad：是否使用AMSGrad变种方法，用于解决Adam算法的收敛性问题。
            调用 torch.optim.Adam 的 step() 方法可以更新模型参数，将优化器计算出的梯度应用于模型参数，并将模型参数更新到下一个迭代。
        """

        scheduler = NoamOpt(
            transformer.hidden_dim, factor=1, warmup=400, optimizer=optimizer,
        )



        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >99% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy = train(
            transformer, scheduler, criterion, batches, masks, n_epochs=n_epochs
        )
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from src.modules.grpo import GRPOLoss


class RLConfig(dict):
    def __getattr__(self, key):
        return self[key]


class GRPOLossTest(unittest.TestCase):
    def test_ratio_clipping_remains_finite_for_large_logprob_delta(self):
        loss_fn = GRPOLoss(
            RLConfig(
                clip_eps=0.2,
                use_latent_loss=True,
                use_answer_loss=False,
                average_per_token_loss=True,
            )
        )
        logprobs = torch.tensor([[1000.0]], requires_grad=True)
        old_logprobs = torch.zeros_like(logprobs)
        attention_mask = torch.ones_like(logprobs)
        advantages = -torch.ones_like(logprobs)

        loss = loss_fn.calculate_loss(
            logprobs=logprobs,
            logprobs_old=old_logprobs,
            attention_mask=attention_mask,
            advantages=advantages,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logprobs.grad).all())


if __name__ == "__main__":
    unittest.main()

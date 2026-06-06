import unittest

from omegaconf import OmegaConf

from run import preprocess_config_hook


def _make_config(do_rl, rl_limit_val_batches=8, trainer_limit_val_batches=None):
    trainer = {
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 2,
    }
    if trainer_limit_val_batches is not None:
        trainer["limit_val_batches"] = trainer_limit_val_batches

    return OmegaConf.create(
        {
            "model": {
                "model_kwargs": {
                    "do_rl": do_rl,
                    "rl_config": {
                        "rl_limit_val_batches": rl_limit_val_batches,
                    },
                },
            },
            "trainer": trainer,
        }
    )


class PreprocessConfigHookTest(unittest.TestCase):
    def test_rl_sets_trainer_limit_val_batches_from_rl_config(self):
        config = _make_config(do_rl=True, rl_limit_val_batches=4)

        processed = preprocess_config_hook(config)

        self.assertEqual(processed.trainer.limit_val_batches, 4)

    def test_rl_keeps_explicit_trainer_limit_val_batches(self):
        config = _make_config(
            do_rl=True,
            rl_limit_val_batches=4,
            trainer_limit_val_batches=2,
        )

        processed = preprocess_config_hook(config)

        self.assertEqual(processed.trainer.limit_val_batches, 2)

    def test_sft_does_not_set_limit_val_batches(self):
        config = _make_config(do_rl=False, rl_limit_val_batches=4)

        processed = preprocess_config_hook(config)

        self.assertNotIn("limit_val_batches", processed.trainer)


if __name__ == "__main__":
    unittest.main()

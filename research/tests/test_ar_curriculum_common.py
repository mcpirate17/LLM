import torch


def test_ar_curriculum_stage_sets_are_shared_by_probe_and_experiment():
    from research.eval import ar_curriculum_probe
    from research.eval._ar_curriculum_common import (
        STAGE_CONFIGS_DEFAULT,
        STAGE_CONFIGS_FINE,
        STAGE_CONFIGS_PROBE,
    )
    from research.tools import ar_curriculum_experiment

    assert ar_curriculum_probe.STAGE_CONFIGS == STAGE_CONFIGS_PROBE
    assert ar_curriculum_experiment.STAGE_SETS["default"] == STAGE_CONFIGS_DEFAULT
    assert ar_curriculum_experiment.STAGE_SETS["fine"] == STAGE_CONFIGS_FINE


def test_ar_curriculum_common_builds_deterministic_stage_batches():
    from research.eval._ar_curriculum_common import (
        STAGE_CONFIGS_PROBE,
        build_stage_specs,
        make_stage_batch,
        required_vocab_size_for_stage_configs,
    )

    device = torch.device("cpu")
    stages = build_stage_specs(7, STAGE_CONFIGS_PROBE[:2], device=device)
    assert len(stages) == 2
    assert stages[0].train_keys.device == device
    assert stages[0].held_keys.shape == (1, 2)
    assert stages[1].value_lo > stages[0].value_hi

    gen = torch.Generator(device=device)
    gen.manual_seed(11)
    ids, targets = make_stage_batch(
        stages[1],
        split="held",
        batch_size=3,
        sep_token=98,
        ans_token=99,
        device=device,
        generator=gen,
    )

    assert ids.shape == (3, 3 * stages[1].pairs_per_example + 4)
    assert targets.shape == (3,)
    assert int(targets.min()) >= stages[1].value_lo
    assert int(targets.max()) < stages[1].value_hi
    assert required_vocab_size_for_stage_configs(STAGE_CONFIGS_PROBE[:2]) == 1036

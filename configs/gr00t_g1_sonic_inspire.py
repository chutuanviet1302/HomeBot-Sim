"""Isaac-GR00T data/modality config for the HomeBot can dataset (G1 + SONIC + Inspire hand).

Mirrors data/homebot_can/meta/modality.json. Register it as a custom embodiment in the
Isaac-GR00T fine-tune script (`--data-config configs/gr00t_g1_sonic_inspire.py:HomeBotConfig`
or by adding it to gr00t/experiment/data_config.py, depending on the repo version).
Check class / transform names against the Isaac-GR00T version you clone (N1.7).
"""
VIDEO_KEYS = ["video.head", "video.wrist"]
STATE_KEYS = ["state.body_joints", "state.right_hand", "state.right_wrist_pos", "state.right_wrist_rot6d"]
ACTION_KEYS = ["action.right_wrist_pos", "action.right_wrist_rot6d", "action.base_height",
               "action.stand", "action.right_hand"]
LANGUAGE_KEYS = ["annotation.human.task_description"]
OBSERVATION_INDICES = [0]
ACTION_HORIZON = 16                    # 0.64 s at 25 fps
ACTION_INDICES = list(range(ACTION_HORIZON))

# Fine-tuning recipe for 2x T4 16 GB (fp16, no bf16 on T4):
TRAIN = dict(
    base_model="nvidia/GR00T-N1.7-3B",   # gated: request access first
    tune_llm=False, tune_visual=False,   # freeze VLM + vision encoder
    tune_projector=True, tune_diffusion_model=True,
    batch_size=4, gradient_accumulation_steps=8,
    learning_rate=1e-4, max_steps=6000, save_steps=500,
    fp16=True, attn_implementation="sdpa",
    embodiment_tag="new_embodiment",
)


def make_config():  # pragma: no cover - requires Isaac-GR00T
    from gr00t.data.dataset import ModalityConfig
    from gr00t.data.transform import (ComposedModalityTransform, StateActionToTensor, StateActionTransform,
                                      VideoResize, VideoToNumpy, VideoToTensor)
    from gr00t.model.transforms import GR00TTransform

    modality = {
        "video": ModalityConfig(delta_indices=OBSERVATION_INDICES, modality_keys=VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=OBSERVATION_INDICES, modality_keys=STATE_KEYS),
        "action": ModalityConfig(delta_indices=ACTION_INDICES, modality_keys=ACTION_KEYS),
        "language": ModalityConfig(delta_indices=OBSERVATION_INDICES, modality_keys=LANGUAGE_KEYS),
    }
    norm = {k: "min_max" for k in STATE_KEYS + ACTION_KEYS}
    transform = ComposedModalityTransform(transforms=[
        VideoToTensor(apply_to=VIDEO_KEYS), VideoResize(apply_to=VIDEO_KEYS, height=224, width=224),
        VideoToNumpy(apply_to=VIDEO_KEYS),
        StateActionToTensor(apply_to=STATE_KEYS), StateActionTransform(apply_to=STATE_KEYS, normalization_modes=norm),
        StateActionToTensor(apply_to=ACTION_KEYS), StateActionTransform(apply_to=ACTION_KEYS, normalization_modes=norm),
        GR00TTransform(state_horizon=1, action_horizon=ACTION_HORIZON, max_state_dim=64, max_action_dim=32),
    ])
    return modality, transform

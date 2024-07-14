from lib.policy_manager import ModeratorPolicyManager

policy_manager = ModeratorPolicyManager()

policy_name="Policy-NakedWoman-Remove"
edited_image_list = policy_manager.call_policy_model(
    policy_name=policy_name,
    prompt="Naked Pretty Woman run on the beach"
)
pretrain_image_list = policy_manager.call_pretrain_model(
    prompt="Naked Pretty Woman run on the beach"
)
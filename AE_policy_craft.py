from lib.policy_manager import ModeratorPolicyManager

policy_manager = ModeratorPolicyManager()

policy_dict = {
    "src_content": {
      "obj": "Naked Woman",
      "sty": "",
      "act": ""
    },
    "method": "REMOVE",
    "purpose": "sexual content",
    "expand_context": "obj",
    "expand_type": "sub-concepts",
}
policy_name = "Naked Woman"
policy_manager.craft_policy(
    policy_dict=policy_dict,
    policy_name=policy_name
)
from build_advantage import build_reasoner_training_signals_batch

if __name__ == '__main__':
    base_rewards_batch = [
        [0, 1, 2, 2, 1, 1],
        [2, 2, 2, 2, 2, 2],
    ]

    comp_scores_batch = [
        [0.1, 0.8, 0.2, 0.9, 0.8, 0.6],
        [0.0, 0.2, 0.8, 0.9, 0.1, 0.4],
    ]

    res = build_reasoner_training_signals_batch(
        base_rewards_batch=base_rewards_batch,
        comp_scores_batch=comp_scores_batch,
        gamma=0.3,
        alpha_pos=0.9,
        alpha_neg=0.9,
    )

    print("a_base_batch =", res["a_base_batch"])
    print(type(res["a_base_batch"][0]))
    print("bonus_batch =", res["bonus_batch"])
    print("final_adv_batch =", res["final_adv_batch"])
    print("-" * 100)
    print("group_info[0] =", res["group_info"][0])
    print("-" * 100)
    print("group_info[1] =", res["group_info"][1])

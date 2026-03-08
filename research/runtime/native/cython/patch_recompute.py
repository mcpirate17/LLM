
with open("/home/tim/Projects/LLM/research/tools/recompute_all_scores.py", "r") as f:
    content = f.read()

repl1 = """               l.quant_int8_retention, l.robustness_long_ctx_score, l.init_sensitivity_std,
               pr.loss_improvement_rate, l.quant_quality_per_byte, l.wikitext_perplexity,
               pr.param_count, pr.most_similar_to, pr.graph_n_params_estimate
        FROM leaderboard l"""

content = content.replace("""               l.quant_int8_retention, l.robustness_long_ctx_score, l.init_sensitivity_std,
               pr.loss_improvement_rate, l.quant_quality_per_byte, l.wikitext_perplexity
        FROM leaderboard l""", repl1)

repl2 = """        (eid, rid, s_lr, s_nov, i_lr, i_rob, v_lr, v_base, v_std, nov_conf, 
         scal_eff, is_ref, rout_sav, comp_ratio, rout_coll, disc_lr, spec_norm, 
         rob_noise, q_ret, long_ctx, init_std, loss_imp, q_qual, w_perp,
         param_count, most_similar_to, graph_n_params) = row"""

content = content.replace("""        (eid, rid, s_lr, s_nov, i_lr, i_rob, v_lr, v_base, v_std, nov_conf, 
         scal_eff, is_ref, rout_sav, comp_ratio, rout_coll, disc_lr, spec_norm, 
         rob_noise, q_ret, long_ctx, init_std, loss_imp, q_qual, w_perp) = row""", repl2)

repl3 = """            quant_quality_per_byte=q_qual,
            wikitext_perplexity=w_perp,
            param_count=param_count,
            most_similar_to=most_similar_to,
            graph_n_params_estimate=graph_n_params
        )"""

content = content.replace("""            quant_quality_per_byte=q_qual,
            wikitext_perplexity=w_perp
        )""", repl3)

with open("/home/tim/Projects/LLM/research/tools/recompute_all_scores.py", "w") as f:
    f.write(content)


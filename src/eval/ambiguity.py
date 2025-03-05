import torch.nn.functional as F
from src.eval.utils import generate_embedding, load_model


def calculate_ambiguity_for_df(df,
                               correct_option_col,
                               option_a_col,
                               option_b_col,
                               option_c_col,
                               option_d_col,
                               ambiguity_col,
                               model_name="BAAI/bge-base-en-v1.5"
):
    model, tokenizer, device = load_model(model_name)

    def ambiguity_score(row):
        options = {
            'a': row[option_a_col], 'b': row[option_b_col],
            'c': row[option_c_col], 'd': row[option_d_col]
        }
        correct_opt = row[correct_option_col].lower()
        if correct_opt not in options:
            return 0
        correct_text = options[correct_opt]
        correct_emb = generate_embedding(correct_text, model, tokenizer, device)
        
        sims = [
            F.cosine_similarity(correct_emb, generate_embedding(text, model, tokenizer, device), dim=0).item()
            for opt, text in options.items() if opt != correct_opt
        ]
        avg_sim = sum(sims) / len(sims) if sims else 0
        return avg_sim

    df[ambiguity_col] = df.apply(ambiguity_score, axis=1)
    return df
import os, glob
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models


def main():
    base_path = './data'
    input_path = os.path.join(base_path, 'input_data')
    output_models_path = os.path.join('./data', 'output_models')
    latest_folder = max(glob.glob(os.path.join(output_models_path, '*/')), key=os.path.getmtime)
    latest_model_path = os.path.join(base_path, latest_folder)

    max_seq_length = 512
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    model.max_seq_length = max_seq_length

    transformer_path = os.path.join(latest_model_path, "0_Transformer")
    word_embedding_model = models.Transformer(transformer_path, max_seq_length=max_seq_length, do_lower_case=True)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    sentence_transformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    test_df = pd.read_csv(os.path.join(input_path, 'trial_1', 'test.csv'), encoding='utf-8-sig')
    test_list = [': '.join([text_1, text_2]) for text_1, text_2 in zip(test_df['title_en'], test_df['description_en'])]
    test_sent_embed = sentence_transformer_model.encode(["hi", "hello"])
    #TODO: output test_list & test_sent_embed to file

if __name__ == "__main__":
    main()

import os
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import evaluation
from sentence_transformers import SentenceTransformer, InputExample, losses, models


def main():
    time_stamp = str(datetime.now().year) + "_" + str(datetime.now().month) + "_" + str(datetime.now().day) + "_" + \
                 str(datetime.now().hour) + "_" + str(datetime.now().minute)
    model_name = "sentence_transformers_model"
    model_dir_data = model_name + "_" + time_stamp
    base_path = os.path.dirname(__file__)
    input_path = os.path.join(base_path, "input_data")
    output_path = os.path.join(base_path, "output_models")
    model_path = os.path.join(output_path, model_dir_data)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    max_seq_length = 512
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    model.max_seq_length = max_seq_length

    train_df = pd.read_csv(os.path.join(input_path, 'trial_1', 'train.csv'), encoding='utf-8-sig')
    train_df['label'] = train_df['label'].astype(float)
    val_df = pd.read_csv(os.path.join(input_path, 'trial_1', 'val.csv'), encoding='utf-8-sig')
    val_df['label'] = val_df['label'].astype(float)
    train_lists = [train_df['title_en'].to_list(), train_df['description_en'].to_list()]
    val_lists = [val_df['title_en'].to_list(), val_df['description_en'].to_list()]

    input_train_examples_list = [InputExample(texts=[text_1, text_2], label=label) for text_1, text_2, label in
                                 zip(train_df['title_en'], train_df['description_en'], train_df['label'])]
    train_dataloader = DataLoader(input_train_examples_list, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    scores = val_df['label'].to_list()
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_df['title_en'].to_list(),
                                                        val_df['description_en'].to_list(), scores)

    def callback(score, epoch, steps):
        pass

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=20, steps_per_epoch=5,
              evaluator=evaluator,
              evaluation_steps=10, output_path=model_path, save_best_model=True,
              optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}, callback=callback(scores, 1, 10))
    transformer_path = os.path.join(model_path, "0_Transformer")
    word_embedding_model = models.Transformer(transformer_path, max_seq_length=max_seq_length, do_lower_case=True)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    sentence_transformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    test_df = pd.read_csv(os.path.join(input_path, 'trial_1', 'test.csv'), encoding='utf-8-sig')
    test_list = [': '.join([text_1, text_2]) for text_1, text_2 in zip(test_df['title_en'], test_df['description_en'])]
    test_sent_embed = sentence_transformer_model.encode(["hi", "hello"])


main()

if __name__ == "__main__":
    main()

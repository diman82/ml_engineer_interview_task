import os, glob
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import evaluation
from sentence_transformers import SentenceTransformer, InputExample, losses, models


def main():
    base_path = '/data'
    input_path = os.path.join(base_path, 'input_data')
    output_models_path = os.path.join(base_path, 'output_models')
    latest_folder = max(glob.glob(os.path.join(output_models_path, '*/')), key=os.path.getmtime)
    latest_model_path = os.path.join(base_path, latest_folder)


    train_df = pd.read_csv(os.path.join(input_path, 'trial_1', 'train.csv'), encoding='utf-8-sig')
    train_df['label'] = train_df['label'].astype(float)
    val_df = pd.read_csv(os.path.join(input_path, 'trial_1', 'val.csv'), encoding='utf-8-sig')
    val_df['label'] = val_df['label'].astype(float)
    train_lists = [train_df['title_en'].to_list(), train_df['description_en'].to_list()]
    val_lists = [val_df['title_en'].to_list(), val_df['description_en'].to_list()]

    max_seq_length = 512
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    model.max_seq_length = max_seq_length

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
              evaluation_steps=10, output_path=latest_model_path, save_best_model=True,
              optimizer_params={'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False}, callback=callback(scores, 1, 10))

if __name__ == "__main__":
    main()

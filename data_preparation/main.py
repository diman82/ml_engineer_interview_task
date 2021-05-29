import os
from datetime import datetime

def main():
    time_stamp = str(datetime.now().year) + "_" + str(datetime.now().month) + "_" + str(datetime.now().day) + "_" + \
                 str(datetime.now().hour) + "_" + str(datetime.now().minute)
    model_name = "sentence_transformers_model"
    model_dir_data = model_name + "_" + time_stamp
    output_path = os.path.join('/data', "output_models")
    model_path = os.path.join(output_path, model_dir_data)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

if __name__ == "__main__":
    main()

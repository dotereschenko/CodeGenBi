import json
import random

# Read the JSON file
with open("/Users/denisteresenko/BICodegen/cosqa/cosqa-retrieval-dev-500.json", "r") as file:
    dataset = json.load(file)


for line in dataset:
    if "negative" not in line:
        doc_strings = [data["docstring_tokens"] for data in dataset if data != line]
        line['negative'] = random.choice(doc_strings)
        with open("new_modified_dataset_dev.json", 'a') as f:
            f.write(json.dumps(line) + '\n')


# with open("new_modified_dataset_train.json", "w") as file:
#     json.dump(dataset, file)
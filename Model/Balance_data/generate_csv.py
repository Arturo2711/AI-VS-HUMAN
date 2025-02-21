import json

def write_line(path, line):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def generate_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            label = data['label']
            text = data['text'].replace('\n', '')
            model = data['model']
            if label and text and model:
                line = f'{label},{model},"{text}"'
                write_line(output_path, line)

if __name__ == "__main__":
    generate_csv('en_train.jsonl', 'data.csv')
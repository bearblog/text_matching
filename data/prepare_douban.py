import os
import ijson
import random
import argparse
import numpy as np
# import tensorflow as tf
from tqdm import tqdm


def parse_config():
    parser = argparse.ArgumentParser()
    param_arg = parser.add_argument_group("Parameters")
    param_arg.add_argument("--min_word_frequency", type=int, default=1,
                           help="Minimum frequency of words in the vocabulary")
    param_arg.add_argument("--max_sentence_len", type=int, default=180, help="Maximum Sentence Length")
    param_arg.add_argument("--random_seed", type=int, default=42, help="Seed for sampling negative training examples")

    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--train_in", type=str, default="orig/Douban_Corpus/train.txt",
                          help="Path to input train file")
    data_arg.add_argument("--train_out", type=str, default="douban/train.txt",
                          help="Path to output train file")
    data_arg.add_argument("--dev_in", type=str, default="orig/Douban_Corpus/valid.txt",
                          help="Path to input dev file")
    data_arg.add_argument("--dev_out", type=str, default="douban/dev.txt",
                          help="Path to output dev file")
    data_arg.add_argument("--test_in", type=str, default="orig/Douban_Corpus/test.txt",
                          help="Path to input dev file")
    data_arg.add_argument("--test_out", type=str, default="douban/test.txt",
                          help="Path to output dev file")
    data_arg.add_argument("--responses_file", type=str, default="orig/Douban_Corpus/responses.txt",
                          help="Path to write answers file")
    data_arg.add_argument("--save_vocab_path", type=str, default="data/Task_1/ubuntu/ubuntu_task_1_vocab.txt",
                          help="Path to save vocabulary file")

    config = parser.parse_args()
    return config


# def to_vec(tokens, vocab, maxlen):
#     '''
#     length: length of the input sequence
#     vec: map the token to the vocab_id, return a varied-length array
#     '''
#     n = len(tokens)
#     length = 0
#     vec=[]
#     for i in range(n):
#         length += 1
#         if tokens[i] in vocab:
#             vec.append(vocab[tokens[i]])
#         else:
#             vec.append(vocab["_OOV_"])
#     return length, np.array(vec)
#
#
# def load_vocab(fname):
#     vocab={}
#     with open(fname, 'rt') as f:
#         for line in f:
#             line = line.strip()
#             fields = line.split('\t')
#             term_id = int(fields[1])
#             vocab[fields[0]] = term_id
#     return vocab


def load_responses(fname):
    responses={}
    with open(fname, 'rt') as f:
        for line in f:
            line = line.strip()
            fields = line.split('\t')
            if len(fields) != 2:
                print("WRONG LINE: {}".format(line))
                r_text = '_OOV_'
            else:
                r_text = fields[1]
            responses[fields[0]] = r_text
    return responses


def create_train_file(file_in, file_out, responses, mode="train"):
    with open(file_out, "w") as file_out_op:
        positive_samples_count = 0
        negative_samples_count = 0

        with open(file_in, 'r') as file_in_op:
            for id, line in enumerate(tqdm(file_in_op.readlines())):
                fields = line.strip().split('\t')
                example_id = fields[0]

                context = fields[1]
                # utterances = (context + ' ').split(' _EOS_ ')[:-1]
                # utterances = [utterance + " _EOS_" for utterance in utterances]
                # utterances = utterances[-max_utter_num:]

                row = ""
                row += str(example_id) + "\t" + context + "\t"

                if fields[2] == "NA" or fields[3] == "NA":
                    continue

                pos_ids = [id for id in fields[2].split('|')]
                for r_id in pos_ids:
                    answer = responses[r_id]
                    correct_answer_row = row + str(r_id) + "\t" + answer + "\t" + "1"
                    file_out_op.write(correct_answer_row.replace("\n", "") + "\n")
                    positive_samples_count += 1

                neg_ids = [id for id in fields[3].split('|')]
                for r_id in neg_ids:
                    answer = responses[r_id]
                    negative_answer_row = row + r_id + "\t" + answer + "\t" + "0"
                    file_out_op.write(negative_answer_row.replace("\n", "") + "\n")
                    negative_samples_count += 1

            print("Saved {} data to {}".format(mode, file_out))
            print("{} - Positive samples count - {}".format(mode, positive_samples_count))
            print("{} - Negative samples count - {}".format(mode, negative_samples_count))


if __name__ == "__main__":
    config = parse_config()
    train_file = os.path.join(config.train_in)
    dev_file = os.path.join(config.dev_in)
    test_file = os.path.join(config.test_in)

    responses_file = os.path.join(config.responses_file)
    # test_answers_file = os.path.join(FLAGS.test_answers_file)

    train_file_out = os.path.join(config.train_out)
    dev_file_out = os.path.join(config.dev_out)
    test_file_out = os.path.join(config.test_out)

    responses = load_responses(responses_file)
    create_train_file(train_file, train_file_out, responses, mode="train")
    create_train_file(dev_file, dev_file_out, responses, mode="dev")
    create_train_file(test_file, test_file_out, responses, mode="test")

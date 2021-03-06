import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import numpy as np
import rouge_not_a_wrapper as my_rouge
import nltk
import json
from tqdm import tqdm
import pickle as pk
import pdb

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote,
              dm_double_close_quote, ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data

extract_sents_num = []
extract_words_num = []
article_sents_num = []

extract_info = {}

nltk.download("punkt")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def chunk_file(set_name):
    in_file = os.path.join(finished_files_dir, '%s.bin' % set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(
            chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack(
                    '%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        if(os.path.exists(os.path.join(finished_files_dir, "%s.bin" % set_name))):
            print("Splitting %s data into chunks..." % set_name)
            chunk_file(set_name)
        else:
            print("Binaries for mode %s are not available, skipping..." % set_name)
    print("Saved chunked data in %s" % chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." %
          (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s),
                                    os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." %
          (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" %
          (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    # print line[-1]
    return line + " ."


def get_art_abs(story_file):
    global article_sents_num
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # get extractive summary
    article_sents = tokenizer.tokenize(article)
    article_sents = [a.encode('utf-8') for a in article_sents]
    extract_sents, extract_ids, fs, ps, rs, max_Rouge_l_r = get_extract_summary(
        article_sents, highlights)

    article_sents_num.append(len(article_sents))
    return article_sents, highlights, extract_sents, extract_ids, fs, ps, rs, max_Rouge_l_r


def get_extract_summary(article_sents, abstract_sents):
    if len(article_sents) == 0 or len(abstract_sents) == 0:
        return [], [], [], [], [], None

    global extract_sents_num
    global extract_words_num
    fscores = []
    precisions = []
    recalls = []
    for i, art_sent in enumerate(article_sents):
        rouge_l_f, rouge_l_p, rouge_l_r = my_rouge.rouge_l_summary_level(
            [art_sent], abstract_sents)
        fscores.append(rouge_l_f)
        precisions.append(rouge_l_p)
        recalls.append(rouge_l_r)

    scores = np.array(recalls)
    sorted_scores = np.sort(scores)[::-1]
    id_sort_by_scores = np.argsort(scores)[::-1]
    max_Rouge_l_r = 0.0
    extract_ids = []
    extract_sents = []
    for i in range(len(article_sents)):
        new_extract_ids = sorted(extract_ids + [id_sort_by_scores[i]])
        new_extract_sents = [article_sents[idx] for idx in new_extract_ids]
        _, _, Rouge_l_r = my_rouge.rouge_l_summary_level(
            new_extract_sents, abstract_sents)
        if Rouge_l_r > max_Rouge_l_r:
            extract_ids = new_extract_ids
            extract_sents = new_extract_sents
            max_Rouge_l_r = Rouge_l_r
    # for those articles that don't reach the 2 conditions
    extract_sents = [bytes_to_str(s) for s in extract_sents]
    if len(extract_sents) == 0:
        pdb.set_trace()
    extract_sents_num.append(len(extract_sents))
    extract_words = " ".join(extract_sents).split(" ")
    extract_words_num.append(len(extract_words))
    return extract_sents, extract_ids, fscores, precisions, recalls, max_Rouge_l_r


def bytes_to_str(bytes_string):
    return str(bytes_string, "utf-8") if isinstance(bytes_string, bytes) else bytes_string


def str_to_bytes(bytes_string):
    return bytes_string.encode("utf-8") if isinstance(bytes_string, str) else bytes_string


def write_to_bin(out_file, lowerBound=0, upperBound=None, makevocab=False):
    upperBound = len(os.listdir(tokenized_stories_dir)) if upperBound == None else upperBound
    fnames = os.listdir(tokenized_stories_dir)
    fnames.sort()
    sliced_dir = fnames[lowerBound:upperBound]
    print("Writing bin for %d inputs" % len(sliced_dir))

    story_fnames = [name for name in sliced_dir]
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    global extract_sents_num
    global extract_words_num
    global article_sents_num
    global extract_info
    extract_sents_num = []
    extract_words_num = []
    article_sents_num = []
    data = {'article': [], 'abstract': [], 'rougeLs': {'f': [], 'p': [], 'r': []},
            'gt_ids': [], 'select_ratio': [], 'rougeL_r': []}

    with open(out_file, 'wb') as writer:
        for idx, s in enumerate(story_fnames):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" %
                      (idx, num_stories, float(idx)*100.0/float(num_stories)))

            # Look in the tokenized story dirs to find the .story file corresponding to this url
            if os.path.isfile(os.path.join(tokenized_stories_dir, s)):
                story_file = os.path.join(tokenized_stories_dir, s)
                print(story_file)

            else:
                print('Error: no data.')

            # Get the strings to write to .bin file
            article_sents, abstract_sents, extract_sents, extract_ids, fs, ps, rs, max_Rouge_l_r = get_art_abs(
                story_file)
            ratio = float(len(extract_sents)) / \
                len(article_sents) if len(article_sents) > 0 else 0

            # save scores of all article sentences
            data['article'].append(article_sents)
            data['abstract'].append(abstract_sents)
            data['rougeLs']['f'].append(fs)
            data['rougeLs']['p'].append(ps)
            data['rougeLs']['r'].append(rs)
            data['gt_ids'].append(extract_ids)
            data['select_ratio'].append(ratio)
            data['rougeL_r'].append(max_Rouge_l_r)
            # Make abstract into a signle string, putting <s> and </s> tags around the sentences
            article = ' '.join(
                ["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in article_sents]
            ).encode("utf-8")

            abstract = ' '.join(
                ["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in abstract_sents]
            ).encode("utf-8")

            extract = ' '.join(
                ["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in extract_sents]
            ).encode("utf-8")

            extract_ids = ','.join([str(i) for i in extract_ids]).encode("utf-8")

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([
                                                                           article])
            tf_example.features.feature['abstract'].bytes_list.value.extend([
                                                                            abstract])
            tf_example.features.feature['extract'].bytes_list.value.extend([
                                                                           extract])
            tf_example.features.feature['extract_ids'].bytes_list.value.extend([
                                                                               extract_ids])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = str(article, "utf-8").split(' ')
                art_tokens = [t for t in art_tokens if t not in [
                    SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                abs_tokens = str(abstract, "utf-8").split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [
                    SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    with open(out_file[:-4] + '_gt.pkl', 'wb') as out:
        pk.dump(data, out)

    print("Finished writing file %s\n" % out_file)
    print("Finished writing file %s\n" % out_file)
    print("average extract sents num: ", float(
        sum(extract_sents_num)) / len(extract_sents_num))
    print("average extract words num: ", float(
        sum(extract_words_num)) / len(extract_words_num))
    print("average article sents num: ", float(
        sum(article_sents_num)) / len(article_sents_num))
    split_name = out_file.split('.')[0]
    extract_info[split_name] = {'extract_sents_num': extract_sents_num,
                                'extract_words_num': extract_words_num,
                                'article_sents_num': article_sents_num}

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("USAGE: python make_datafiles.py <stories_dir> <out_dir> <test_rate>")
        sys.exit()

    stories_dir = sys.argv[1]
    out_dir = sys.argv[2]
    validation_rate = float(sys.argv[3])

    # print(stories_dir)
    # tokenized_stories_dir = "data/tokenized_stories_dir"
    # finished_files_dir = "data/finished_files"
    tokenized_stories_dir = os.path.join(out_dir, "tokenized_stories_dir")
    finished_files_dir = os.path.join(out_dir, "finished_files")

    chunks_dir = os.path.join(finished_files_dir, "chunked")

    # Create some new directories
    if not os.path.exists(tokenized_stories_dir):
        os.makedirs(tokenized_stories_dir)
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_stories(stories_dir, tokenized_stories_dir)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    upBound = len(os.listdir(tokenized_stories_dir))
    sepIdx = round(validation_rate * upBound)
    halfOfRest = int(round((upBound - sepIdx)/2))
    write_to_bin(os.path.join(finished_files_dir, "train.bin"),
                 lowerBound=0, upperBound=sepIdx, makevocab=True)
    write_to_bin(os.path.join(finished_files_dir, "val.bin"),
                 lowerBound=sepIdx, upperBound=sepIdx+halfOfRest)
    write_to_bin(os.path.join(finished_files_dir, "test.bin"),
                 lowerBound=sepIdx+halfOfRest, upperBound=upBound)

    # Extractive summary
    with open(os.path.join(finished_files_dir, 'extract_info.pkl'), 'wb') as output_file:
        pk.dump(extract_info, output_file)

    # # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()

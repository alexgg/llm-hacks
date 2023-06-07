import os
import pprint as pp
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding
import sys
from getopt import getopt
from getopt import GetoptError

# The input data needs to be structured as;
#
# {DELIMITER}HEADER
# BODY
# {DELIMITER} HEADER
#
# The header and body get combined into a single combined text that is then
# encoded into a single vector embedding
#

DELIMITER='---'
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002
MAX_TOKENS = 8000  # the maximum for text-embedding-ada-002 is 8191
SEPARATOR='|'
openai.api_key = os.environ['OPENAI_API_KEY']

def process(filepath):
    ''' Process the input file into a list of [header, body] elements '''
    header=None
    body=[]
    data=[]
    with open(filepath, mode='r', encoding='utf-8') as ifile:
        for line in ifile:
            if line.isspace() or len(line) == 0:
                continue
            if line.startswith(DELIMITER):
                if header and body:
                    data.append([header, '\n'.join(body)])
                    body = []
                header = line.rstrip()
                body=[]
            else:
                body.append(line.rstrip())

    if header and body:
        data.append([header, '\n'.join(body)])
    return data


def process_embedding(data):
    ''' Process the input data list into a dataFrame with embeddings '''
    data_frame=pd.DataFrame(data, columns=['header', 'body'])
    data_frame.dropna()
    data_frame['combined'] = ("Header: " + data_frame['header'] + "; Body: " +
                              data_frame['body'])

    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    # omit bodies that are too long to embed
    data_frame["n_tokens"] = data_frame.combined.apply(lambda x: len(encoding.encode(x)))
    data_frame = data_frame[data_frame.n_tokens <= MAX_TOKENS]
    data_frame['embedding'] = data_frame.combined.apply(
        lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    return data_frame


def main(argv):
    try:
        opts, args = getopt(argv[1:], "hvi:o:", ["help", "verbose=",
                                                 "input-file=",
                                                 "output-file="])
    except GetoptError as ex:
        error("get opt error: %s" % (str(ex)))
        usage()

    arg_dict = {}
    arg_dict['verbose']=None
    arg_dict['input-file']=None
    arg_dict['output-file']=None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
        elif o in ("-v", "--verbose"):
            arg_dict["verbose"] = True
        elif o in ("-i", "--input-file"):
            arg_dict["input-file"] = str(a)
        elif o in ("-o", "--output-file"):
            arg_dict["output-file"] = str(a)
        else:
            assert False, "unhandled option"

    try:
        if arg_dict["verbose"] is not None:
            print(f"""Arguments:\t{arg_dict}""")

        if arg_dict['input-file'] is None or arg_dict['output-file'] is None:
            print("Both input-file and output-file are required")
            usage()
            sys.exit(1)

        data_frame = process_embedding(process(arg_dict['input-file']))
        with open(arg_dict['output-file'], mode='w', encoding='utf-8') as ofile:
            data_frame.to_csv(ofile, sep=SEPARATOR, encoding='utf-8',
                              header='false', index='false',  mode='w',
                              escapechar='\\')
    except KeyError:
        pass


def error(message):
    ''' Print error '''
    print ("Error: " + message)

def usage():
    ''' Print usage '''
    print ('Usage: " + script_name + [OPTIONS]:\nOptions:\n\
           \t-h Display usage\n \
           \t-p User prompt\n \
           \t-i Input file\n \
           \t-o Output file path\n \
           \t-v Verbose output\n')
    sys.exit(1)

if __name__ == '__main__':
    script_name = sys.argv[0]
    main(sys.argv)

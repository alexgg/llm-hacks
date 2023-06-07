#!/usr/bin/env python3

import sys, os
from getopt import getopt
from getopt import GetoptError
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

openai.api_key = os.environ['OPENAI_API_KEY']

DELIMITER = "####"

def get_completion(system_message, prompt, model="gpt-3.5-turbo", assistant_message='None'):
    assert openai.api_key, 'Please provide an API key in the OPENAI_API_KEY environment variable'
    ret = openai.Moderation.create(
        prompt
    )
    moderated_output = ret.results[0]
    if moderated_output['flagged'] !='true':
        # Use DELIMITER to avoid promp injection
        user_message = prompt.replace(DELIMITER, "")
        user_message = f"""\
        {DELIMITER}{user_message}{DELIMITER}
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": assistant_message},
            {"role": "user", "content": user_message}
        ]
        ret = openai.ChatCompletion.create(
            model = model,
            messages=messages,
            temperature=0
        )
        return ret.choices[0].message["content"]


def search(data_frame, query, n_rows=1):
    ''' Search for similarities using the embeddings '''
    embedding = get_embedding(query, engine='text-embedding-ada-002')
    data_frame.head()
    data_frame["embedding"] = data_frame.embedding.apply(eval).apply(np.array)
    data_frame['similarities'] = data_frame.embedding.apply(
        lambda x: cosine_similarity(x, embedding))

    res = data_frame.sort_values('similarities', ascending=False).head(n_rows)
    return res


def main(argv):
    try:
        opts, args = getopt(argv[1:], "hvp:s:a:m:e:", ["help", "verbose=", "prompt=","system-message=", "assistant-message=", "model=", "--embeddings"])
    except GetoptError as ex:
        error("get opt error: %s" % (str(ex)))
        usage()

    arg_dict = {}
    arg_dict['system']=""
    arg_dict['prompt']=""
    arg_dict['assistant']=""
    arg_dict['verbose']=None
    arg_dict['embeddings-file']=None
    arg_dict['model']="gpt-3.5-turbo"
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
        elif o in ("-v", "--verbose"):
            arg_dict["verbose"] = True
        elif o in ("-p", "--prompt"):
            arg_dict["prompt"] = str(a)
        elif o in ("-s", "--system-message"):
            arg_dict["system"] = str(a)
        elif o in ("-a", "--assistant-message"):
            arg_dict["assistant"] = str(a)
        elif o in ("-m", "--model"):
            arg_dict["model"] = str(a)
        elif o in ("-e", "--embeddings"):
            arg_dict["embeddings-file"] = str(a)
        else:
            assert False, "unhandled option"

    try:
        if arg_dict["verbose"] is not None:
            print(f"""Arguments:\t{arg_dict}""")

        if not openai.api_key:
            print("Please provide an openAI API key by setting the OPENAI_API_KEY environment variable")
            sys.exit(1)

        context=None
        if arg_dict["embeddings-file"] is not None:
            data_frame = pd.read_csv(arg_dict['embeddings-file'], index_col=0, sep='|')
            results = search(data_frame, arg_dict['prompt'], n_rows=3)
            context = results['body'].str.cat(sep=' ')
            if arg_dict['verbose']:
                print(f"""Additional context: {context}""")

        system_message = f"""message will be delimited with {DELIMITER} \
characters. Use the extra context provided next: {context}. {arg_dict['system']}"""
        response = get_completion(system_message, arg_dict['prompt'], arg_dict['model'])
        print(response)
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
           \t-s System message\n \
           \t-a Assistant message\n \
           \t-m GPT model (defaults to gpt-3.5-turbo)\n \
           \t-v Verbose output\n')
    sys.exit(1)

if __name__ == '__main__':
    script_name = sys.argv[0]
    main(sys.argv)

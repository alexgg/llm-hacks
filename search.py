import sys
import os
import pprint as pp
from getopt import getopt
from getopt import GetoptError
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
#from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

openai.api_key = os.environ['OPENAI_API_KEY']

def search(data_frame, query, n_rows=1):
    ''' Search for similarities using the embeddings '''
    embedding = get_embedding(query, engine='text-embedding-ada-002')
    data_frame.head()
    data_frame["embedding"] = data_frame.embedding.apply(eval).apply(np.array)
    data_frame['similarities'] = data_frame.embedding.apply(
        lambda x: cosine_similarity(x, embedding))

    res = data_frame.sort_values('similarities', ascending=False).head(n_rows)
    return res

#def create_db(token, data_frame):
#    embedding_function = OpenAIEmbeddingFunction(api_key=token, model_name=EMBEDDING_MODEL)
#    combined_collection = chroma_client.create_collection(name='combined_contents', embedding_function=embedding_function)
#    combined_collection.add(
#        ids=data_frame.id.tolist(),
#        embeddings=data_frame.combined.tolist(),
#    )


def main(argv):
    try:
        opts, args = getopt(argv[1:], "hve:q:", ["help", "verbose=", "embeddings-file=", "query="])
    except GetoptError as ex:
        error("get opt error: %s" % (str(ex)))
        usage()

    arg_dict = {}
    arg_dict['verbose']=None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
        elif o in ("-v", "--verbose"):
            arg_dict["verbose"] = True
        elif o in ("-e", "--embeddings-file"):
            arg_dict["embeddings-file"] = str(a)
        elif o in ("-q", "--query"):
            arg_dict["query"] = str(a)
        else:
            assert False, "unhandled option"

    try:
        if arg_dict["verbose"] is not None:
            print(f"""Arguments:\t{arg_dict}""")

        data_frame = pd.read_csv(arg_dict['embeddings-file'], index_col=0, sep='|')
        results = search(data_frame, arg_dict['query'])
        highest_similarity = results.similarities.max()
        if highest_similarity >= 0.85:
            print(data_frame.loc[data_frame.similarities == highest_similarity, 'body'])
        else:
            # TODO: Should call out to openai.Completion.create for chatGPT API
            print("Sorry, I don't know the answer")

    except KeyError:
        pass


def error(message):
    ''' Print error '''
    print ("Error: " + message)

def usage():
    ''' Print usage '''
    print ('Usage: " + script_name + [OPTIONS]:\nOptions:\n\
           \t-h Display usage\n \
           \t-e Input embeddings file\n \
           \t-q Query\n \
           \t-v Verbose output\n')
    sys.exit(1)

if __name__ == '__main__':
    script_name = sys.argv[0]
    main(sys.argv)

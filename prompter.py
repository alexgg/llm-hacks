#!/usr/bin/env python3

import sys, os
from getopt import getopt
from getopt import GetoptError
import openai

openai.api_key = os.environ['OPENAI_API_KEY']

DELIMITER = "####"

def get_completion(system_message, prompt, assistant_message='None', model="gpt-3.5-turbo"):
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


def main(argv):
    try:
        opts, args = getopt(argv[1:], "hvp:s:a:", ["help", "verbose=", "prompt=","system-message=", "assistant-message="])
    except GetoptError as ex:
        error("get opt error: %s" % (str(ex)))
        usage()

    arg_dict = {}
    arg_dict['system']=""
    arg_dict['prompt']=""
    arg_dict['assistant']=""
    arg_dict['verbose']=None
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
        else:
            assert False, "unhandled option"

    try:
        if arg_dict["verbose"] is not None:
            print(f"""Arguments:\t{arg_dict}""")

        system_message = f"""message will be delimited with {DELIMITER} \
characters.{arg_dict['system']}"""
        response = get_completion(system_message, arg_dict['prompt'])
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
           \t-v Verbose output\n')
    sys.exit(1)

if __name__ == '__main__':
    script_name = sys.argv[0]
    main(sys.argv)
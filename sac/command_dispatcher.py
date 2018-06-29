import os
import sys
import subprocess
import re
import argparse


name = 'test'
command = 'echo "hello world"'
command_list = [
    'source ~/projects/venv/bin/activate',
    'python main.py --run-name=test1 --num-columns=1 --distance-mode=mean',
    '`while true; do sleep 10000; done`', # hax to prevent the screen from dying if something goes wrong.
]

def run_in_screen(screen_name, commands, venv_path=None, wait_for_interrupt=False):
    pre_commands = [] if venv_path is None else [f'source {venv_path}']
    post_commands = [] if not wait_for_interrupt else ['`while true; do sleep 10000; done`']
    command_list = pre_commands + commands + post_commands
    processed_commands = '; '.join(command_list)
    return subprocess.call(f'screen -S {screen_name} -dm bash -c \'{processed_commands}\'', shell=True)

class FileProcessingException(Exception):
    pass

def process_file_line(line):
    line = line.strip()

    match = re.match(r'^(\S*?)\s*\:\s*([\s\S]*)$', line)
    if not match:
        raise FileProcessingException(f'Could not read line: {line}')
    (screen_name, command) = match.groups()
    screen_name = screen_name.strip()
    command = command.strip()
    return screen_name, command

def dispatch(file_path, venv_path=None, wait_for_interrupt=False):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    screen_name_command_pairs = []
    for i, line in enumerate(lines):
        try:
            screen_name_command_pairs.append(process_file_line(line))
        except FileProcessingException as e:
            raise Exception(f'Failed to read line {i+1}: "{line}"')
    for screen_name, command in screen_name_command_pairs:
        run_in_screen(screen_name, [command], venv_path=venv_path, wait_for_interrupt=wait_for_interrupt)


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--venv', type=str, default=None)
parser.add_argument('--dont-wait', action='store_true')
args = parser.parse_args()

dispatch(args.file, venv_path=args.venv, wait_for_interrupt=(not args.dont_wait))
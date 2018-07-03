import os
import sys
import subprocess
import re
import argparse

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

def dispatch(file_path, venv_path=None, wait_for_interrupt=False, load_tensorboard=False):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        file_name = f.name
    screen_name_command_pairs = []
    for i, line in enumerate(lines):
        try:
            screen_name_command_pairs.append(process_file_line(line))
        except FileProcessingException as e:
            raise Exception(f'Failed to read line {i+1}: "{line}"')
    for screen_name, command in screen_name_command_pairs:
        run_in_screen(screen_name, [command], venv_path=venv_path, wait_for_interrupt=wait_for_interrupt)
    if load_tensorboard:
        tb_command = produce_tensorboard_command(run_dir='runs', common_path='data',
                                                 run_names=[screen_name for screen_name, _ in screen_name_command_pairs])
        run_in_screen(f'tb_{file_name}', [tb_command], venv_path=venv_path, wait_for_interrupt=wait_for_interrupt)

def produce_tensorboard_command(run_dir, common_path, run_names):
    run_paths = [f'{run_name}:{os.path.join(run_dir, run_name, common_path)}' for run_name in run_names]
    return f"tensorboard --logdir={','.join(run_paths)}"






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--venv', type=str, default=None)
    parser.add_argument('--dont-wait', action='store_true')
    parser.add_argument('--tb-command', type=str, default=None)
    args = parser.parse_args()
    if args.tb_command is None:
        if args.file is None:
            raise Exception(f'Must specify --file flag.')
        dispatch(args.file, venv_path=args.venv, wait_for_interrupt=(not args.dont_wait))
    else:
        evaluated_run_list = eval(args.tb_command)
        print(produce_tensorboard_command('runs', 'data', evaluated_run_list))

#print(produce_tensorboard_command('runs', 'data', [f'nc_future{i+1}' for i in range(0, 6)] + \
#                                                  [f'nc_future256_{i+1}' for i in range(0, 6)] + \
#                                                  [f'nc_future_10_{i+1}' for i in range(0, 6)] ))
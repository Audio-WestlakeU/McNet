import argparse
import fnmatch
import os
from os.path import join, exists
import shutil
import yaml
import re


def to_smaller_log(oldfile: str, newfile: str):
    print("shorten", oldfile, os.stat(oldfile).st_size / (1024 * 1024), 'MB')
    with open(oldfile, 'r') as f:
        nlines = []
        while True:
            l = f.readline()
            if not l:
                break
            if l.startswith("\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K"):
                nlines = nlines[:-4]
                nlines.append(l.replace("\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K\x1b[1A\x1b[2K", ""))
            elif l.startswith("\x1b[2K\x1b[1A\x1b[2K"):
                nlines = nlines[:-2]
                nlines.append(l.replace("\x1b[2K\x1b[1A\x1b[2K", ""))
            elif l.startswith("\x1b[2K"):
                nlines = nlines[:-1]
                nlines.append(l.replace('\x1b[2K', ''))
            else:
                nlines.append(l.replace('\x1b[?25l', '').replace('\x1b[?25h', ''))
            if len(nlines) > 10000:
                print("over 10000 lines, ignore", oldfile)
                f.close()
                return
        f.close()

        with open(newfile, 'a') as f2:
            for l in nlines:
                print(l.replace("\n", ""))
                f2.write(l)
            f2.close()
            os.remove(oldfile)

def move(alg: str, max_ver: int):
    if max_ver < 0:
        return
    versions = [f"version_{v}" for v in range(max_ver)]
    for v in versions:
        v_log = os.path.join(f"logs/{alg}", v + '.log')
        v_log_dest = os.path.join(f"logs/{alg}", v, v + '.log')
        if os.path.exists(v_log):
            to_smaller_log(v_log, v_log_dest)


def move_task_logs(task_dir: str = 'tasks'):

    def move_task_log_to(log: str, dir: str):
        with open(log, 'r') as f:
            v_num = None
            while True:
                l = f.readline()
                if not l:
                    break
                m = re.match(".* v_num: (\d+) ", l)
                if m:
                    v_num = m.group(1)
                    f.close()
                    break
            print('\n\n\nmove', log, join(dir, f'version_{v_num}', f'version_{v_num}.log'))
            to_smaller_log(log, join(dir, f'version_{v_num}', f'version_{v_num}.log'))

    if exists(task_dir) and exists('logs'):
        configs = fnmatch.filter(os.listdir(join(task_dir, 'done')), '*.yaml')
        for c in configs:
            log = join(task_dir, 'done', c + '.log')
            cfile = join(task_dir, 'done', c)
            if not exists(log):
                continue
            with open(cfile, 'r') as f:
                config = yaml.load(f, yaml.SafeLoader)
                cmd = config['cmd']
                if "NBSSCLI.py" in cmd:
                    move_task_log_to(log, dir='logs/NBSS')
                elif "models.BeamGuidedTasNet" in cmd:
                    move_task_log_to(log, dir='logs/BeamGuidedTasNet')
                elif "models.FaSNet_TAC" in cmd:
                    move_task_log_to(log, dir='logs/FaSNet_TAC')
                elif "SepFormer" in cmd:
                    move_task_log_to(log, dir='logs/SepFormer')
                else:
                    print('ignore', log)


if __name__ == "__main__":
    # Example: python utils/test_all_ovlps.py --fnsf 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--fnsf', type=int, default=-1, help='需要移动log文件的最大version号')
    args = parser.parse_args()

    # move_task_logs()

    move('FreqNarrSubFullNet', args.fnsf)

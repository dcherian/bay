#!/usr/bin/python3

import os
import glob
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('action', help='action to take', type=str,
                    choices=['from-ganges',
                             'to-ganges',
                             'git',
                             'local-git',
                             'rebase-ebob-rama',
                             'recombine',
                             'server-to-ganges'])
parser.add_argument('unit', help='specific unit or deployment to target e.g. 527 ebob',
                    default=[], nargs=argparse.REMAINDER)
parser.add_argument('server', help='specify matlab server e.g. 1',
                    default=2, nargs=argparse.REMAINDER)

args = parser.parse_args()

cmdrsync = ['rsync', '-ahtP']  # base rsync command

ganges = '/media/ganges/data/'
local = '/home/deepak/'
mserver = '/home/server/pi/homes/dcherian/'

# directories on ganges
gdirs = ['chipod/RAMA13/', 'chipod/RAMA14/', 'EBoB13/']
# directories on local machine
ldirs = ['rama/RAMA13/', 'rama/RAMA14/', 'ebob/']

# directories on MATLAB server
mdirs = ['RAMA13/', 'RAMA14/', 'ebob13/']

# base "project" branch for instrument
unitBranch = ['rama', 'rama', 'ebob']
# prefix for unit-specific branch
unitBranchPrefix = ['RAMA13', 'RAMA14', 'ebob']

allunits = dict()
allunits['RAMA13'] = ('526', '527')
allunits['RAMA14'] = ('810', '811', '812', '813', '814')
allunits['RAMA'] = ('526', '527', '810', '811', '812', '813', '814')
allunits['EBOB'] = ('500', '501', '504', '505', '511', '514', '516', '518', '519')

newunits = []
for uu in args.unit:
    if uu.upper() in allunits.keys():
        [newunits.append(aa) for aa in allunits[uu.upper()]]
    else:
        newunits.append(uu)

args.unit = newunits


def RebaseEbobRama():
    podpath = '/home/deepak/chipod_gust/'
    cdcmd = 'cd ' + podpath

    filename = podpath+'/mfiles/chipod_gust/driver/combine_turbulence.m'
    emacseval = '\'(progn (find-file \"' + filename + '\") (magit-status))\''

    for unitBranch in ['ebob', 'rama']:
        print('--- rebasing ' + unitBranch + ' onto master')

        subprocess.run(cdcmd
                       + ' && git checkout ' + unitBranch + ' -q',
                       shell=True, check=True)

        # get commit if rebase is necessary
        mergebase = subprocess.run(cdcmd
                                   + ' && git merge-base master'
                                   + ' ' + unitBranch,
                                   stdout=subprocess.PIPE,
                                   shell=True)
        rebasefrom = mergebase.stdout[:8].decode()

        try:
            # git rebase --onto rama COMMIT^
            subprocess.run(cdcmd
                           + ' && git rebase --onto master'
                           + ' ' + rebasefrom,
                           shell=True, check=True)
        except subprocess.CalledProcessError:
            # run magit on file
            print('=======')
            print('Rebase failed. Opening ' + filename)
            print('=======')
            subprocess.run('emacsclient -c --eval '
                           + emacseval, shell=True)

        subprocess.run(cdcmd + ' && git push --force-with-lease '
                       + 'origin -q',
                       shell=True, check=True)

    subprocess.run(cdcmd + ' && git checkout master -q', shell=True, check=True)


def SyncGit(podpath, branch, prefix, kind='ganges'):

    if 'sal-gap' in podpath:
        print('Skipping 511-sal-gap')
        return

    cdcmd = 'cd '+podpath+'/mfiles/chipod_gust/'
    filename = podpath+'/mfiles/chipod_gust/driver/combine_turbulence.m'
    emacseval = '\'(progn (find-file \"' + filename + '\") (magit-status))\''

    unitBranch = prefix + '-' + podname

    # update remotes
    subprocess.run(cdcmd+' && git fetch -q', shell=True, check=True)

    # get commit if rebase is necessary
    mergebase = subprocess.run(cdcmd
                               + ' && git merge-base ' + branch
                               + ' ' + unitBranch,
                               stdout=subprocess.PIPE,
                               shell=True)
    rebasefrom = mergebase.stdout[:8].decode()
    # print('Merge base is ' + rebasefrom)

    print('--- resetting to origin/master')
    try:
        subprocess.run(cdcmd
                       + ' && git checkout master -q'
                       + ' && git reset --hard origin/master -q',
                       shell=True, check=True)
    except subprocess.CalledProcessError:
        subprocess.run(cdcmd + ' && '
                       + 'git commit -a -m "ganges-pod: automatic commit."',
                       shell=True, check=True)

        subprocess.run(cdcmd
                       + ' && git checkout master -q'
                       + ' && git pull origin -q',
                       shell=True, check=True)

    print('--- resetting to ' + branch)
    subprocess.run(cdcmd
                   + ' && git checkout '+branch+' -q'
                   + ' && git reset --hard origin/'+branch+' -q',
                   shell=True, check=True)

    # checkout unit branch
    subprocess.run(cdcmd
                   + ' && git checkout '+unitBranch+' -q',
                   shell=True, check=True)

    # check if deployment branch is ancestor of unit branch
    # if so, no rebase necessary
    isancestor = subprocess.run(cdcmd
                                + ' && git merge-base --is-ancestor '
                                + branch + ' ' + unitBranch,
                                shell=True)

    if isancestor.returncode != 0:
        print('--- rebasing ' + unitBranch + ' onto ' + branch)

        try:
            # git rebase --onto rama COMMIT^
            subprocess.run(cdcmd
                           + ' && git rebase --onto '+branch
                           + ' ' + rebasefrom,
                           shell=True, check=True)
        except subprocess.CalledProcessError:
            # run magit on file
            print('=======')
            print('Rebase failed. Opening ' + filename)
            print('=======')
            subprocess.run('emacsclient -c --eval '
                           + emacseval, shell=True)
            subprocess.run(cdcmd + ' && git push --set-upstream --force-with-lease '
                           + 'origin ' + unitBranch,
                           shell=True, check=True)
    else:
        print(unitBranch + ' is up-to-date! No rebase required.')


def CopyFromGanges(podpath):

    pp = podpath

    for dirs in ['calib', 'input', 'pics', 'mfiles']:
        localdir = lpath + podname + '/' + dirs + ''

        if not os.path.isdir(localdir):
            print('... making dir: ' + localdir)
            subprocess.run(['mkdir', '-p', localdir])

        command = list(cmdrsync)
        command.append(pp + '/' + dirs + '/')
        command.append(localdir)
        print('... ' + dirs)
        subprocess.run(command)

    # proc is special
    print('... proc')
    command = list(cmdrsync)
    for ff in ['Turb*', 'P*', 'T_m', 'temp*']:
        remotefile = pp + '/proc/' + ff + '.mat'
        command.append(remotefile)
        command.append(lpath + podname + '/proc/')
        subprocess.run(' '.join(command), shell=True)

    remotefile = pp + '/proc/chi/*.mat'
    command = list(cmdrsync)
    command.append(remotefile)
    command.append(lpath + podname + '/proc/chi/')
    subprocess.run(' '.join(command), shell=True)

    remotefile = pp + '/proc/chi/stats/*.mat'
    command = list(cmdrsync)
    command.append(remotefile)
    command.append(lpath + podname + '/proc/chi/stats/')
    subprocess.run(' '.join(command), shell=True)

    # files to delete
    # for ff in ['temp', 'motion']:
    #     fname = lpath + podname + '/proc/' + ff + '.mat'
    #     if os.path.isfile(fname):
    #         print('..... deleting ' + fname)
    #         subprocess.run(['rm', fname])

    command = list(cmdrsync)
    if os.path.isdir(pp + '/proc/combined'):
        command.append(pp + '/proc/combined')
        command.append(lpath + podname + '/proc/')
        subprocess.run(' '.join(command), shell=True)


def CopyToGanges(podpath):

    pp = podpath

    for dirs in ['calib', 'input', 'pics']:
        localdir = lpath + podname + '/' + dirs + '/'

        if not os.path.isdir(localdir):
            return

        command = list(cmdrsync)
        command.append(localdir)
        command.append(pp + '/' + dirs + '/')
        print('... ' + dirs)
        subprocess.run(command)


def Recombine(podpath, podname, gpath, mpath, server):

    print('recombining...')

    # ganges → MATLAB server
    print('copying to MATLAB server ' + server + '...')
    for dirs in ['calib', 'input', 'mfiles', 'pics']:
        command = list(cmdrsync)
        command.append(podpath + '/' + dirs)
        command.append('matlab' + server + ':' + mpath + podname + '/')
        print(command)
        subprocess.run(command, check=True)

    # submit job on MATLAB server
    command = ['ssh', 'matlab'+server,
               '\'cd ' + mpath + podname + '/mfiles/' +
               ' && sh run.sh combine_turbulence.m\'']
    subprocess.run(' '.join(command), shell=True, check=True)


def ServerToGanges(podpath, podname, mpath):

    # MATLAB server → ganges
    print('copying from MATLAB server...')
    for dirs in ['pics', 'proc']:
        command = list(cmdrsync)
        command.append('--update')
        command.append('matlab4:' + mpath
                       + podname + '/' + dirs)
        command.append(podpath + '/')
        print(command)
        subprocess.run(command, check=True)


if args.action == 'rebase-ebob-rama':
    RebaseEbobRama()
    exit()

# actually loop
for dd, ll, mm, br, prefix in zip(gdirs, ldirs, mdirs,
                                  unitBranch, unitBranchPrefix):
    gpath = ganges + dd + 'data/'
    lpath = local + ll + 'data/'
    mpath = mserver + mm + 'data/'

    if args.action != 'local-git':
        if not os.path.isdir(gpath):
            raise FileExistsError('ganges is not mounted!')

        pods = glob.glob(gpath + '/*')

    else:
        pods = glob.glob(lpath + '/*')

    for pp in pods:
        if os.path.isdir(pp):
            podname = os.path.split(pp)[-1]
            if (args.unit == [] or (podname in args.unit)) and podname != '503':
                print('\n =======> Doing ' + podname)

                if args.action == 'from-ganges':
                    CopyFromGanges(pp)

                if args.action == 'to-ganges':
                    CopyToGanges(pp)

                if args.action == 'git' or args.action == 'local-git':
                    SyncGit(pp, br, prefix)

                if args.action == 'recombine':
                    Recombine(pp, podname, gpath, mpath, args.server)

                if args.action == 'server-to-ganges':
                    ServerToGanges(pp, podname, mpath)

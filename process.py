from glob import glob
from os.path import join, splitext, basename
from multiprocessing import Process, Pool
import multiprocessing, re, os, shutil, tqdm

dbroot = '/hdd1/home/thkim/data/IEMOCAP_full_release' 
savepath = join(dbroot, 'preprocessed')
llist = sorted(glob(join(dbroot, 'Session*/dialog/EmoEvaluation/*.txt')))
alist = sorted(glob(join(dbroot, 'Session*/sentences/wav/*/*.wav')))
vlist = sorted(glob(join(dbroot, 'Session*/dialog/avi/DivX/*.avi')))
videocutpath = join(savepath, 'timing.txt')
allowed_emo = {'Neutral': 0, 'neu':0, 'Anger': 1, 'ang': 1, 
        'Happiness':2, 'hap': 2, 'Sadness': 3, 'sad': 3}
gen_converter = {'F': 0, 'M':1}

MAX_DURATION = 6.

def label_preprocessing(labelname):
    with open(labelname, 'r') as f:
        emo_detailed_treat = True
        imp_or_scr, fname, start, duration = '', '', '', ''
        for line in f:
            line = line.strip()
            if line.startswith('['):
                global fname, imp_or_scr, start, duration
                time, fname, emo, avd = line.split('\t')
                start, end = map(float, time[1:-1].split(' - '))
                duration = MAX_DURATION if end-start > MAX_DURATION else end-start
                imp_or_scr = fname.split('_')[1]
                gen = fname.split('_')[-1][0]
                gen_int = str(gen_converter[gen])
                
                if not os.path.exists(join(savepath, fname)):
                    os.makedirs(join(savepath, fname))
                else:
                    shutil.rmtree(join(savepath, fname))
                    os.makedirs(join(savepath, fname))

                with open(join(savepath, fname, fname + '.avd'), 'w') as g:
                    g.write(avd[1:-1])
                with open(join(savepath, fname, fname + '.gen'), 'w') as g:
                    g.write(gen_int)
                if emo in allowed_emo.keys():
                    with open(join(savepath, fname, fname + '.cat'), 'w') as g:
                        g.write('{:d}\n'.format(allowed_emo[emo]))
                    emo_detailed_treat = False

            # This part takes consideration on cat ambiguity
            elif imp_or_scr.startswith('script') and line.startswith('C-E') and emo_detailed_treat: 
                clist = [xx.strip(';') for xx in line.split('\t')[1].split()]
                for citem in clist:
                    if citem in allowed_emo.keys():
                        with open(join(savepath, fname, fname + '.cat'), 'a') as g:
                            g.write('{:d}\n'.format(allowed_emo[citem]))

            elif imp_or_scr.startswith('impro') and line.startswith('C') and emo_detailed_treat: 
                if not line.startswith('C-{}'.format(gen)):
                    continue
                clist = [xx.strip(';') for xx in line.split('\t')[1].split()]
                for citem in clist:
                    if citem in allowed_emo.keys():
                        with open(join(savepath, fname, fname + '.cat'), 'a') as g:
                            g.write('{:d}\n'.format(allowed_emo[citem]))
            elif line=='' and not fname=='':
                emo_detailed_treat = True
                if not os.path.exists(join(savepath, fname, fname + '.cat')):
                    os.system('rm -rf {}'.format(join(savepath, fname)))
                else:
                    with open(videocutpath, 'a') as g:
                        g.write('{}\t{}\t{:.3f}\t{:.3f}\n'.format(rawname(labelname), fname, start, duration ))

def trim_videos():
    with open(videocutpath, 'r') as f:
        lines = [xx.strip() for xx in f.readlines()]
    p = Pool()
    p.map_async(trim_video, lines)
    p.close(); p.join()

def trim_video(line):
    inname, outname, start, end = line.split('\t')
    sn = inname[4]
    outname = join(savepath, outname, outname)
    inname = join(dbroot, 'Session%s/dialog/avi/DivX'%sn, inname)
    trim_command = ffmpeg(inname,outname, float(start), float(end))
    os.system(trim_command)

def extract_avs():
    flist = sorted(glob(join(savepath,'*','*.avi')))
    p = Pool()
    p.map_async(extract_av, flist)
    p.close(); p.join()

def extract_av(fname):
    fname = fname[:-4]
    gen = fname[-4]
    os.system(ffmpeg_extract_audio(fname))
    os.system(ffmpeg_extract_frames(fname, gen))

def rawname(filename):
    return basename(splitext(filename)[0])

def ffmpeg(inname, outname, start, duration):
    return 'ffmpeg -i {}.avi -ss {} -t {} -acodec copy -y -v debug {}.avi 2> {}.trim.log'.format(
            inname, sec2hms(start), sec2hms(duration), outname, outname)

def ffmpeg_extract_audio(inname):
    return 'ffmpeg -i {}.avi -vn -ar 16000 -y -v debug {}.wav 2> {}.aud.log'.format(inname, inname, inname)

def ffmpeg_extract_frames(inname, gen):
    if gen=='M':
        w, h, x, y = 224, 224, 438, 128
    elif gen=='F':
        w, h, x, y = 224, 224, 87, 128
    else:
        raise ValueError('invalid gen, I got {}'.format(gen))
    return 'ffmpeg -i {}.avi -vf "crop={}:{}:{}:{},scale=96:96" -y -r 25 -v debug {}_%05d.jpg 2> {}.fra.log'.format(inname, w, h, x, y, inname, inname)

def sec2hms(sec):
    h = int(sec//3600)
    m = int((sec%3600)//60)
    s = int(sec%60)
    ms = int((sec%60 - s) * 1000)
    return '{:02d}:{:02d}:{:02d}.{:03d}'.format(h, m, s, ms)

if __name__=='__main__':
    if os.path.exists(videocutpath):
        os.system('rm {}'.format(videocutpath))
    for lname in tqdm.tqdm(llist):
        label_preprocessing(lname)
    print("Done labelling")
    trim_videos()
    print("Done trimming")
    extract_avs()
    print("Done extracting")

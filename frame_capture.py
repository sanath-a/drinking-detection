import numpy as np
import pandas as pd
import os, shutil
import cv2
import math
import re


def Extractor(file_names, verbose = True):
    '''
    Extracts frames from videos and separates into training, testing, and validation
    sets with a 50/30/20 split.

    ---Params---
    file_names: Names of video files for extraction
    verbose: Print statements on/off
    ------------
    ---Returns---
    Nothing! Creates File Repositories
    --------------
    '''
    names = []
    for f in file_names:
        name, _ = os.path.splitext(f)
        names.append(name)

    paths = ['./data/videos/' + f for f in file_names]

    for n in names:
        if not os.path.isdir('./data' + n + '/'):
            os.mkdir('./data/' + n + '/')

    total = 0
    for (p, n) in zip(paths, names):
        num_done = extractImages(p, './data/' + n + '/')
        total += num_done
        if verbose: print (n + ": %d Frames Extracted" % num_done)
    if verbose: print ("%d Total Frames Collected" % total)

def SetBuilder(path = './data/'):
    '''
    Builds training, validation and test sets from image dump. Assumes Extractor
    has been run. Builds masked and original data sets.

    ---Params---
    Path: Path to img dump from Extractor
    ------------
    ---Returns---
    Void. Creates file repositories for datasets.
    -------------
    '''
    og_vids = []
    masked_vids = []
    all_files = []
    for root, dirnames, filenames in os.walk(path):
        file_paths = [os.path.join(root, f) for f in filenames if os.path.splitext(f)[1] == '.jpg']
        idx = [re.sub('[abcdefghijklmnopqrstuv_/.S]', '', f) for f in file_paths]
        idx = np.argsort([int(i) for i in idx])
        file_paths = [file_paths[i] for i in idx]
        all_files += file_paths
        if 'orginal' in root:
            og_vids += file_paths
        else:
            masked_vids += file_paths


    annotated_df = pd.DataFrame(columns = ['File', 'Target'])
    names = [f.replace('./data/','') for f in all_files]
    names = [f.replace('/', '_') for f in names]

    annotated_df['File'] = names
    annotated_df.to_csv('annotated.csv', sep = ',', header = False, index = False)
    for f in (og_vids, masked_vids):
        shuffled_files = np.random.permutation(f)

        half_idx = int(math.ceil(len(f)/2))
        eight_idx = int(math.floor(len(f) * 0.8))

        train_files = shuffled_files[:half_idx]
        test_files = shuffled_files[half_idx:eight_idx]
        val_files = shuffled_files[eight_idx:]

        sets = [train_files, test_files, val_files]
        if 'orginal' in f[0]:
            dirs = ['./data/original/train/', './data/original/test/', './data/original/val/']
        else:
            dirs = ['./data/masked/train/', './data/masked/test/', './data/masked/val/']



        for d in dirs:
            if not os.path.isdir(d):
                os.makedirs(d)

        for s, d in zip(sets, dirs):
            for f in s:
                basename = f.replace('./data/', '')
                basename = basename.replace('/', '_')
                shutil.copyfile(f, d + basename)
                os.chmod(d + basename, 0o0777)



def CleanUp(path = './data/', keep_sets = False):
    dirs = os.listdir(path)
    for d in dirs:
        if d in ['original', 'masked'] and keep_sets:
            continue
        if 'videos' not in d and os.path.isdir(path + d):
            shutil.rmtree(path + d)
    if not keep_sets:
        os.remove('./annotated.csv')

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    frames_in_vid = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    while count < frames_in_vid:
      vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
      success,image = vidcap.read()
      cv2.imwrite( pathOut + "frame%d.jpg" % count, image)
      os.chmod(pathOut + "frame%d.jpg" % count, 0o0777)
      count = count + 1
    return count

if __name__ == "__main__":
    videos = [f for f in os.listdir('./data/videos/') if os.path.splitext(f)[1] == '.mp4']
    print ("Cleaning Old Files...")
    #CleanUp()
    print ("Extracting Frames...")
    #Extractor(videos)
    print ("Building Datasets...")
    #SetBuilder()
    print ("Cleaning data dumps...")
    #CleanUp(keep_sets = True)
    print ("Frame Extraction: Success")

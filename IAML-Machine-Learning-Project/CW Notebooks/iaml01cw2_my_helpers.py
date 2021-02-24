##########################################################
#  Python module template for helper functions of your own (IAML Level 10)
#  Note that:
#  - Those helper functions of your own for Questions 1, 2, and 3 should be defined in this file.
#  - You can decide function names by yourself.
#  - You do not need to include this header in your submission
##########################################################

import os
import gzip
import numpy as np
import iaml01cw2_helpers

def load_language_data():
    file_path = os.path.join(os.getcwd(),"data")
    Xtrn, Ytrn, Xtst, Ytst = iaml01cw2_helpers.load_CoVoST2(file_path)
    label_path = os.path.join(os.getcwd(),"data","languages.txt")
    languages = open(label_path,'r').read().split('\n')[:-1]
    language_name = [language[3:].split(" ")[0] for language in languages]
    language_label = [str(i) for i in range(22)]
    return Xtrn,Ytrn,Xtst,Ytst,language_name,language_label

def normalize_image_data():
    file_path = os.path.join(os.getcwd(),"data")
    Xtrn, Ytrn, Xtst, Ytst = iaml01cw2_helpers.load_FashionMNIST(file_path)
    Xtrn_orig, Xtst_orig = Xtrn, Xtst
    Xtrn, Xtst = Xtrn / 255.0, Xtst / 255.0
    Xmean = np.mean(Xtrn,axis=0)
    Xtrn_nm, Xtst_nm = Xtrn - Xmean, Xtst - Xmean
    return Xmean,Xtrn,Ytrn,Xtst,Ytst,Xtrn_nm,Xtst_nm

    
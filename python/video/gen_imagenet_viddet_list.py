"""This script reads all the annotations from some video dataset and generates 
an input list file which can be used by the input data layer
"""

from __future__ import print_function
import logging
import os
import os.path
import sys
import argparse
import traceback
import glob
import numpy as np
import cv2
from collections import defaultdict
import xml.etree.ElementTree as ET


def write_annos_to_file(all_annos, out_filepath):
    """Write output file list which can be read by the caffe's video data layer
    """
    fd = open(out_filepath, 'w+')
    # iterate over all videos
    for vid_filepattern in all_annos:
        c_vid = all_annos[vid_filepattern]
        fd.write("%s %d\n" % (vid_filepattern, len(c_vid)))
        # iterate over all objects in this video
        for obj_track in c_vid:
            obj_trackid = obj_track[0]
            obj_name = obj_track[1]
            fd.write("%s %s %d\n" % (obj_trackid, obj_name, len(c_vid[obj_track])))
            # iterate over all bounding boxes for this object
            for bb in c_vid[obj_track]:
                fd.write("%d %d %d %d %d %d\n" % tuple(bb))
    fd.close()


def disp_bb(input_filepattern, num_frames, bndboxs):
    """Visualizes all bounding boxes over all frames for a video sequence
    """
    # iterate over all the frames
    for frame_no in range(num_frames):
        img = cv2.imread(input_filepattern % frame_no)
        # iterate over all the objects
        for obji, obj_track in enumerate(bndboxs.keys()):
            # check if this object has an annotation for this frame
            bb = [bb for bb in bndboxs[obj_track] if bb[0] == frame_no]
            assert len(bb) <= 1, "Single object has multiple annnotations per frame"
            if bb:
                bb = bb[0]
                # get color of this object
                if len(bndboxs) == 1:
                    clrrange = 0
                else:
                    clrrange = obji / (len(bndboxs) - 1)
                r = 255 * clrrange
                g = 255 * (1 - clrrange)
                b = 255 * (1 - clrrange)
                cv2.rectangle(img, (bb[1], bb[2]), (bb[3], bb[4]), (r, g, b), 2)
                cv2.putText(img,'occl' if bb[5] else '',(bb[1], bb[3]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255))

        cv2.imshow('frame', img)
        cv2.waitKey(0)
    

def imagenet_vid_processing(inputdir, outputfile, extra_args):
    """Generates the dataset list for imagenet video detection
    """
    IM_EXT = ".JPEG"
    IM_FILEPATTERN = "%06d" + IM_EXT
    ANNO_EXT = ".xml"
    ANNO_FILEPATTERN = "%06d" + ANNO_EXT

    # check if the annotation and data directory exists
    imagesetsdir = os.path.join(inputdir, "ImageSets/VID")
    if not os.path.isdir(imagesetsdir):
        raise Exception("Directory '%s' doesn't exist" % imagesetsdir)

    # figure out what set type you want to look at
    if extra_args.settype:
        settype = extra_args.settype
    else:
        settype = "train_1.txt"
    set_choices = ["train", "test", "val"]
    selected_choice = [settype.startswith(x) for x in set_choices]
    if not any(selected_choice):
        raise Exception("'%s' set type not supported for imagenet" % settype)
    # find whether train, test, or val needed
    tmp = [i for i,tmp in enumerate(selected_choice) if tmp]
    settype_major = set_choices[tmp[0]]

    datadir = os.path.join(inputdir, "Data/VID/" + settype_major)
    annodir = os.path.join(inputdir, "Annotations/VID/" + settype_major)
    if not os.path.isdir(datadir):
        raise Exception("Directory '%s' doesn't exist" % datadir)

    # the dictionary that stores all annotations
    # its a double dictionary of lists
    # - the top level key is a string identifier for the video
    # - the second level key is an integer identifier for the object id
    # - each all_annos['video_id'][obj_num] is a list of annotations
    all_annos = {}

    imagesetfiles = glob.glob(os.path.join(imagesetsdir, settype + "*"))
    # iterate over all imagelist files
    for imagesetf in imagesetfiles:
        fd = open(imagesetf, 'r')
        lines = fd.readlines()
        fd.close()
        # iterate over all the video directories in this list
        for line in lines:
            currviddir = line.split()[0]
            # check if both input directory and annotation directory exists
            c_datadir = os.path.join(datadir, currviddir)
            c_annodir = os.path.join(annodir, currviddir)
            if not os.path.isdir(c_datadir):
                raise Exception("Directory '%s' doesn't exist" % c_datadir)
            if not os.path.isdir(c_annodir):
                raise Exception("Directory '%s' doesn't exist" % c_annodir)
            # get number of frames in this videos
            c_imfiles = glob.glob(os.path.join(c_datadir, "*" + IM_EXT))
            c_annofiles = glob.glob(os.path.join(c_annodir, "*" + ANNO_EXT))
            if len(c_imfiles) != len(c_annofiles):
                raise Exception("The number of annotation and images not same for '%s'" % currviddir[0])

            vid_filepattern = os.path.join(c_datadir, IM_FILEPATTERN)
            # initialize annotation for this video (initialized with defaultdict
            # so its easier to append value to the list of an object)
            all_annos[vid_filepattern] = defaultdict(list)

            # read each annotation file
            for annofile in c_annofiles:
                frameno = os.path.splitext(os.path.basename(annofile))[0]
                frameno = int(frameno)
                tree = ET.parse(annofile)
                root = tree.getroot()
                for obj_anno in root.findall("object"):
                    obj_trackid = obj_anno.find("trackid").text
                    obj_name = obj_anno.find("name").text
                    obj_bndbox = obj_anno.find("bndbox")
                    obj_xmax = int(obj_bndbox.find("xmax").text)
                    obj_xmin = int(obj_bndbox.find("xmin").text)
                    obj_ymax = int(obj_bndbox.find("ymax").text)
                    obj_ymin = int(obj_bndbox.find("ymin").text)
                    obj_occl = int(obj_anno.find("occluded").text)
                    # print(obj_trackid, obj_name, obj_xmax, obj_xmin, obj_ymax, obj_ymin, obj_occl)
                    all_annos[vid_filepattern][(obj_trackid, obj_name)].append( \
                        [frameno, obj_xmin, obj_ymin, obj_xmax, obj_ymax, obj_occl])

            # sort all object lists by frame numbers
            for obj_track in all_annos[vid_filepattern]:
                all_annos[vid_filepattern][obj_track].sort(key=lambda elem: elem[0])

            # disp_bb(vid_filepattern, len(c_imfiles), all_annos[vid_filepattern])

            print("%s -> %d objs" % (vid_filepattern, len(all_annos[vid_filepattern])))

    # return all the annotations obtained
    return all_annos


def segtrack_processing(inputdir, outputfile, extra_args):
    """Generates the dataset list for Segtract
    """
    raise Exception("SegTrack processing not yet implemented")


class SmartFormatter(argparse.HelpFormatter):
    """Used for formatting argparse strings by prefacing the help text with R|
    """
    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('R|'):
            txts = text[2:].splitlines()
            rtxts = []
            for txt in txts:
                rtxts += argparse.HelpFormatter._split_lines(self, txt, width)
            return rtxts
        return argparse.HelpFormatter._split_lines(self, text, width)


# Main entry function, where user arguments are handled. According to the
# user arguments different functionality is invoked. Also we initialize the
# logger from this function and catch all exceptions
if __name__ == "__main__":
    # set the logger
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)
    _hdlr = logging.StreamHandler()
    _formatter = logging.Formatter('%(name)-12s: %(levelname)-8s - %(message)s')
    _hdlr.setFormatter(_formatter)
    _hdlr.setLevel(logging.DEBUG)
    logger.addHandler(_hdlr)

    parser = argparse.ArgumentParser(description=\
"------------------  Video Dataset Annotation List Generator  ------------------" \
"-------------------------------------------------------------------------------" \
"This script reads all the annotations from some video dataset and generates " \
"an input list file which can be used by the input data layer ", \
formatter_class=SmartFormatter)

    rpgroup = parser.add_argument_group('required named arguments')
    rpgroup.add_argument("-i", "--inputdir", type=str, metavar="INPUTDIR", \
                         help="Input directory for the dataset", required=True)
    rpgroup.add_argument("-d", "--datasettype", choices=["imagenet", "segtrack"], \
                         help="R|Indicates the type of the dataset. The possible\n" \
                              "arguments are:\n" \
                              "[imagenet]: processes imagenet video detection dataset\n" \
                              "[segtrack]: processes GATech Segtrack dataset", \
                              required=True)
    rpgroup.add_argument("-s", "--settype", type=str, metavar="SETTYPE", \
                         help="R|Indicates the sub-type of the dataset. The choices\n" \
                              "depend on the dataset you use.")
    rpgroup.add_argument("-o", "--outputfile", type=str, metavar="OUTPUTFILE", \
                         help="Output filepath", required=True)

    args = parser.parse_args()

    # iterate over all options - and just execute according to first option
    try:
        inputdir = os.path.realpath(args.inputdir)
        if not os.path.isdir(inputdir):
            raise Exception("Directory '%s' doesn't exist" % inputdir)

        outputfile = os.path.realpath(args.outputfile)
        if args.datasettype == "imagenet":
            all_annos = imagenet_vid_processing(inputdir, args.outputfile, args)
            write_annos_to_file(all_annos, outputfile)
        elif args.datasettype == "segtrack":
            all_annos = segtrack_processing(inputdir, args.outputfile, args)
            write_annos_to_file(all_annos, outputfile)
        else:
            # if help needed
            parser.print_help()

    except Exception as _err:
        logger.error(_err)
        traceback.print_exc(file=sys.stderr)
        sys.exit(-2)

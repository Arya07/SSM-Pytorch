import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
# import utils.cython_bbox
import pickle as cPickle
import subprocess
import uuid
from datasets.icub_eval import icub_eval
from model.config import cfg
#from confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt

class icub_dataset(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'iCWT_' + '_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = self._devkit_path

        self._classes = self._load_classes()

        #self._classes = ('__background__', # always index 0
        #                 'cellphone1', 'mouse2', 'perfume1','remote4', 'soapdispenser1', 'sunglasses5',  'glass8', 'hairbrush4', 'ovenglove1', 'squeezer5')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        assert os.path.exists(self._devkit_path), \
                'iCubWorld-Translation_devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'data Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Images', #index contiene la riga dell'imageset
                                  index + self._image_ext)   #file scelto quindi nel nostro caso class/instance/..../0002152
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_classes(self):
        """
        Load the classes list
        """
        classes_list_file = os.path.join(self._data_path, 'Classes', self._image_set + '_classes.txt')
        assert os.path.exists(classes_list_file), 'Path does not exist: {}'.format(classes_list_file)

        with open(classes_list_file) as f:                                         
            classes_list = [x.strip() for x in f.readlines()]
        classes = tuple(classes_list)

        return classes


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /data/ImageSets/train.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:                                         #da cambiare
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where iCubWorld-Translation_devkit is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'datasets', 'iCWT', 'iCubWorld-Transformations')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_icub_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb


######################## begin #############################
    def replace_gt(self, ss_candidate, ss_fake_gt, flip):
        '''replace gt with ss_fake_gt'''
        gt = self.gt_roidb()
        print('length of gt roidb:{}'.format(len(gt)))
        if flip:
            self._image_index=self._image_index[:len(gt)]
        # replace some gt by fake_gt[fake_idx]
        for j,i in enumerate(ss_candidate):
            if (flip and i<len(gt)) or not flip:
                gt[i]=ss_fake_gt[j]
        if cfg.TRAIN.PROPOSAL_METHOD=='selective_search':
            ss_roidb = self._load_selective_search_roidb(gt)
            roidb = datasets.imdb.merge_roidbs(gt, ss_roidb)
        else:
            roidb = gt
        print('replace gt with self pace gt')
        self._roidb = roidb
        return roidb
######################## end #############################


    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_icub_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        #with open('prova.txt', 'wt') as f:
            # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            #x1 = float(bbox.find('xmin').text)
            #y1 = float(bbox.find('ymin').text)
            #x2 = float(bbox.find('xmax').text)
            #y2 = float(bbox.find('ymax').text)
    #        print str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' '+ str(y2) + ' '

    #        f.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' '+ str(y2) + ' \n')

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_icub_results_file_template(self):
        # iCubWorld-Translation_devkit/results/_det_train.txt
        # filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filename = '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _get_icub_results_file_template_classification(self):
        # iCubWorld-Translation_devkit/results/_det_train.txt
        # filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filename = '_classification_' + self._image_set + '.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_icub_results_file(self, all_boxes):
        #results file for detection
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} icub_dataset results file'.format(cls))
            filename = self._get_icub_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the icubdevkit expects 0-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))
        #results file for classification
        results_file_name = self._get_icub_results_file_template_classification()
        with open(results_file_name, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                max_score = 0.0
                max_cls = ''
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    print(cls_ind)
                    if not dets.size:
                        continue
                    # print dets
                    argmx_vec = np.argmax(dets, axis=0)
                    new_possible_max = dets[argmx_vec[-1]][-1]
                    if max_score < new_possible_max:
                        max_det = dets[argmx_vec[-1], :]
                        max_cls = cls
                f.write('{:s} {:s}\n'.format(index, max_cls))



    def _do_python_eval(self, output_dir = 'output', store=False):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        precs = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(output_dir + '/mAP.txt', 'wt') as f1:
            for i, cls in enumerate(self._classes):
                if cls == '__background__':
                    continue
                filename = self._get_icub_results_file_template().format(cls)
                rec, prec, ap = icub_eval(
                    filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
                aps += [ap]
                precs += [prec]
                print('AP for {} = {:.4f}'.format(cls, ap))
                f1.write('AP for {} = {:.4f}\n'.format(cls, ap))
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            f1.write('Mean AP = {:.4f}'.format(np.mean(aps)))

        #plot_confusion_matrix(self.classes[1:], self._get_icub_results_file_template_classification())
        print(aps)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

        if store:
            file_name = output_dir + '/results.npy'
            try:
                res = np.load(file_name)
                res = np.hstack((res, np.mean(aps)))
                np.save(file_name, res)
                print('Updated old results file')
            except:
                res = np.mean(aps)
                np.save(file_name, res)
                print('Results file not found. Created a new one and stored')

    #
    # def _do_matlab_eval(self, output_dir='output'):
    #     print '-----------------------------------------------------'
    #     print 'Computing results with the official MATLAB eval code.'
    #     print '-----------------------------------------------------'
    #     path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
    #                         'VOCdevkit-matlab-wrapper')
    #     cmd = 'cd {} && '.format(path)
    #     cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    #     cmd += '-r "dbstop if error; '
    #     cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
    #            .format(self._devkit_path, self._get_comp_id(),
    #                    self._image_set, output_dir)
    #     print('Running:\n{}'.format(cmd))
    #     status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, store=False):
        self._write_icub_results_file(all_boxes)
        self._do_python_eval(output_dir, store=store)
        # if self.config['matlab_eval']:
        #     self._do_matlab_eval(output_dir)
        # if self.config['cleanup']:
        #     for cls in self._classes:
        #         if cls == '__background__':
        #             continue
        #         filename = self._get_voc_results_file_template().format(cls)
        #         os.remove(filename)
    #
    # def competition_mode(self, on):
    #     if on:
    #         self.config['use_salt'] = False
    #         self.config['cleanup'] = False
    #     else:
    #         self.config['use_salt'] = True
    #         self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.icub_dataset import icub_dataset
    d = icub_dataset('train') # nome dell'imageset file
    res = d.roidb
    from IPython import embed; embed()

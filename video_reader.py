import torch
from torchvision import transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
from transformations.augmentation import KineticsResizedCropFewshot, ColorJitter, Compose
from transformations.random_erasing import RandomErasing
import torchvision.transforms._transforms_video as transforms
from numpy.random import randint

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []
    
    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        # random_idx = np.random.choice(match_idxs)
        random_idx = random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)

"""Dataset for few-shot videos, which returns few-shot tasks. """
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args, train_mode):

        self.args = args

        self.get_item_counter = 0

        self.data_dir = args.path
        self.seq_len = args.seq_len
        self.train = train_mode
        
        # self.tensor_transform = transforms.ToTensor()
        self.img_size = args.img_size

        self.annotation_path = args.traintestlist

        self.way=args.way
        self.shot=args.shot
        self.query_per_class=args.query_per_class

        self.train_split = Split()
        self.test_split = Split()

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
    ############transform_train################
        std_transform_list_query = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range = [256,256],
                    crop_size = 224,
                ),]
        std_transform_list = [
                            transforms.ToTensorVideo(),
                            KineticsResizedCropFewshot(
                                short_side_range = [256, 256],
                                crop_size = 224,
                            ),
                            # transforms.RandomHorizontalFlipVideo()
                        ]
        std_transform_list_query.append(
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5,
                hue = 0.25,
                grayscale = 0.3,
                consistent = True,
                shuffle = True,
                gray_first = True,
                is_split = False
            ),
        )
        std_transform_list_query += [
                        transforms.NormalizeVideo(
                            mean=[0.45, 0.45, 0.45],
                            std = [0.225, 0.225, 0.225],
                            inplace=True
                        ),
                        RandomErasing()
                        ]
        std_transform_list += [
                            transforms.NormalizeVideo(
                                mean= [0.45, 0.45, 0.45],
                                std=  [0.225, 0.225, 0.225],
                                inplace=True
                            ),
                        ]                
    ############transform_train################

    ############transform_test################ 
        resize_video = KineticsResizedCropFewshot(
                            short_side_range = [256,256], #[256, 256]
                            crop_size = 224, #224
                            num_spatial_crops = 1, #1
                            idx = True
                        )   # KineticsResizedCrop
        std_transform_list_test = [
            transforms.ToTensorVideo(),
            resize_video,
            transforms.NormalizeVideo(
                mean=[0.485, 0.456, 0.406], #[0.45, 0.45, 0.45]
                std=[0.229, 0.224, 0.225], #[0.225, 0.225, 0.225]
                inplace=True
            )
        ]           
         
        self.transform = {}
        self.transform["train_support"] = Compose(std_transform_list)
        self.transform["train_query"] = Compose(std_transform_list_query)
        self.transform["test"] = Compose(std_transform_list_test)

    
    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """
    def read_dir(self):
        # load zipfile into memory
        if self.data_dir.endswith('.zip'):
            self.zip = True
            zip_fn = os.path.join(self.data_dir)
            self.mem = open(zip_fn, 'rb').read()
            self.zfile = zipfile.ZipFile(io.BytesIO(self.mem))
        else:
            self.zip = False

        # go through zip and populate splits with frame locations and action groundtruths
        if self.zip:
            dir_list = list(set([x for x in self.zfile.namelist() if '.jpg' not in x]))

            class_folders = list(set([x.split(os.sep)[-3] for x in dir_list if len(x.split(os.sep)) > 2]))
            class_folders.sort()
            self.class_folders = class_folders
            video_folders = list(set([x.split(os.sep)[-2] for x in dir_list if len(x.split(os.sep)) > 3]))
            video_folders.sort()
            self.video_folders = video_folders

            class_folders_indexes = {v: k for k, v in enumerate(self.class_folders)}
            video_folders_indexes = {v: k for k, v in enumerate(self.video_folders)}
            
            img_list = [x for x in self.zfile.namelist() if '.jpg' in x]
            img_list.sort()

            c = self.get_train_or_test_db(video_folders[0])

            last_video_folder = None
            last_video_class = -1
            insert_frames = []
            for img_path in img_list:
            
                class_folder, video_folder, jpg = img_path.split(os.sep)[-3:]

                if video_folder != last_video_folder:
                    if len(insert_frames) >= self.seq_len:
                        c = self.get_train_or_test_db(last_video_folder.lower())
                        if c != None:
                            c.add_vid(insert_frames, last_video_class)
                        else:
                            pass
                    insert_frames = []
                    class_id = class_folders_indexes[class_folder]
                    vid_id = video_folders_indexes[video_folder]
               
                insert_frames.append(img_path)
                last_video_folder = video_folder
                last_video_class = class_id

            c = self.get_train_or_test_db(last_video_folder)
            if c != None and len(insert_frames) >= self.seq_len:
                c.add_vid(insert_frames, last_video_class)
        else:
            class_folders = os.listdir(self.data_dir)
            class_folders.sort()
            self.class_folders = class_folders
            for class_folder in class_folders:
                video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                video_folders.sort()
                if self.args.debug_loader:
                    video_folders = video_folders[0:1]
                for video_folder in video_folders:
                    c = self.get_train_or_test_db(video_folder)
                    if c == None:
                        continue
                    imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder))
                    if len(imgs) < self.seq_len:
                        continue            
                    imgs.sort()
                    paths = [os.path.join(self.data_dir, class_folder, video_folder, img) for img in imgs]
                    paths.sort()
                    class_id =  class_folders.index(class_folder)
                    c.add_vid(paths, class_id)
        print("loaded {}".format(self.data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    """ return the current split being used """
    def get_train_or_test_db(self, split=None):
        if split is None:
            get_train_split = self.train
        else:
            if split in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"]:
                get_train_split = False
            else:
                return None
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
    
    """ load the paths of all videos in the train and test splits. """ 
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, self.args.split)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.replace(' ', '_') for x in data]
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                
                if "kinetics" in self.args.path:
                    data = [x[0:11] for x in data]
                
                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)
   
    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes
    
    """Loads a single image from a specified path """
    def read_single_image(self, path):
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i
    
    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        num_segments = 8
        seg_length =  1 
        total_length = num_segments * seg_length

        n_frames = len(paths)
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            num_frames = n_frames
            offsets = list()
            ticks = [i * num_frames // num_segments for i in range(num_segments + 1)]
            for i in range(num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= seg_length:
                    tick += randint(tick_len - seg_length + 1)
                offsets.extend([j for j in range(tick, tick + seg_length)])
            idxs =  offsets

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train_support"]
            else:
                transform = self.transform["test"]
            imgs = [torch.tensor(np.array(v)) for v in imgs]
            imgs = torch.stack(imgs)
            imgs = transform(imgs)
            imgs = imgs.transpose(1,0)
        return imgs, vid_id

    def get_seq_query(self, label, idx=-1):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx) 
        n_frames = len(paths)
        num_segments = 8
        seg_length =  1 
        total_length = num_segments * seg_length
        num_frames = n_frames
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]

        else:
            offset = (num_frames / num_segments - seg_length) / 2.0
            out = np.array([i * num_frames / num_segments + offset + j
                         for i in range(num_segments)
                         for j in range(seg_length)], dtype=np.int)
            idxs = [int(f) for f in out]
       
        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train_query"]
            else:
                transform = self.transform["test"]
            imgs = [torch.tensor(np.array(v)) for v in imgs]
            imgs = torch.stack(imgs)
           
            imgs = transform(imgs)
            imgs = imgs.transpose(1,0)
        return imgs, vid_id, idxs


    """returns dict of support and target images and labels"""
    def __getitem__(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []
        ids_list =[]
        for bl, bc in enumerate(batch_classes):
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)
            
            for idx in idxs[0:self.args.shot]:
                vid, vid_id = self.get_seq(bc, idx)
                # vid, vid_id, ids = self.get_seq_query(bc, idx)
                support_set.append(vid)
                support_labels.append(bl)
                real_support_labels.append(bc)

            for idx in idxs[self.args.shot:]:
                vid, vid_id, ids = self.get_seq_query(bc, idx)
                ids_list.append(ids)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)
        s = list(zip(support_set, support_labels, real_support_labels))
        random.shuffle(s)
        support_set, support_labels, real_support_labels = zip(*s)
        
        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)
        
        support_set = torch.cat(support_set)

        target_set = torch.cat(target_set)

        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        real_support_labels = torch.FloatTensor(real_support_labels)
        batch_classes = torch.FloatTensor(batch_classes) 
        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, "real_support_labels":real_support_labels ,"real_target_labels":real_target_labels, "batch_class_list": batch_classes}



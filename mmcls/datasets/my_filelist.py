import mmcv
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class MyFilelist(BaseDataset):
    CLASSES = [
        'Covid',
        'Normal',
        'Penumoina'
    ]

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                if filename.startswith('Pneumonia-Bacterial'):
                    str_list=list(filename)
                    b=' '
                    str_list.insert(len('Pneumonia-Bacterial'),b)
                    filename="".join(str_list)

                if filename.startswith('Pneumonia-Viral'):
                    str_list=list(filename)
                    b=' '
                    str_list.insert(len('Pneumonia-Viral'),b)
                    filename="".join(str_list)
                
                if filename.startswith('Viral'):
                    str_list=list(filename)
                    b=' '
                    str_list.insert(len('Viral'),b)
                    filename="".join(str_list)
                
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos
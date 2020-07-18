import numpy as np
from src.utils.config import cfg
from pathlib import Path
from src.dataset.base_dataset import BaseDataset
import re
import urllib


inst_list = ['bur', 'chr', 'els', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']

class QAPLIB(BaseDataset):
    def __init__(self, sets, obj_resize, cls, fetch_online=False):
        super(QAPLIB, self).__init__()
        self.classes = ['qaplib']
        self.obj_resize = obj_resize
        self.sets = sets

        if cls is not None and cls != 'none':
            idx = inst_list.index(cls)
            self.inst_list = [inst_list[idx]]
        else:
            self.inst_list = inst_list

        self.data_list = []
        self.qap_path = Path(cfg.QAPLIB.DIR)
        for inst in self.inst_list:
            for dat_path in self.qap_path.glob(inst + '*.dat'):
                name = dat_path.name[:-4]
                prob_size = int(re.findall(r"\d+", name)[0])
                if prob_size > 200 or (self.sets == 'train' and prob_size >= 100):
                    continue
                self.data_list.append(name)

        self.data_list.sort()

        #if sets == 'train':
        #    self.data_list = ['tho40']

        fetched_flag = self.qap_path / 'fetched_online'

        if fetch_online or not fetched_flag.exists():
            self.__fetch_online()
            fetched_flag.touch()

    def get_pair(self, idx, shuffle=None):
        """
        Get QAP data by index
        :param idx: dataset index
        :param shuffle: no use here
        :return: (pair of data, groundtruth permutation matrix)
        """
        name = self.data_list[idx]

        dat_path = self.qap_path / (name + '.dat')
        sln_path = self.qap_path / (name + '.sln')
        dat_file = dat_path.open()
        sln_file = sln_path.open()

        def split_line(x):
            for _ in re.split(r'[,\s]', x.rstrip('\n')):
                if _ == "":
                    continue
                else:
                    yield int(_)

        dat_list = [[_ for _ in split_line(line)] for line in dat_file]
        sln_list = [[_ for _ in split_line(line)] for line in sln_file]

        prob_size = dat_list[0][0]

        # read data
        r = 0
        c = 0
        Fi = [[]]
        Fj = [[]]
        F = Fi
        for l in dat_list[1:]:
            F[r] += l
            c += len(l)
            assert c <= prob_size
            if c == prob_size:
                r += 1
                if r < prob_size:
                    F.append([])
                    c = 0
                else:
                    F = Fj
                    r = 0
                    c = 0
        Fi = np.array(Fi, dtype=np.float32)
        Fj = np.array(Fj, dtype=np.float32)
        assert Fi.shape == Fj.shape == (prob_size, prob_size)
        #K = np.kron(Fj, Fi)

        # read solution
        sol = sln_list[0][1]
        perm_list = []
        for _ in sln_list[1:]:
            perm_list += _
        assert len(perm_list) == prob_size
        perm_mat = np.zeros((prob_size, prob_size), dtype=np.float32)
        for r, c in enumerate(perm_list):
            perm_mat[r, c - 1] = 1

        return Fi, Fj, perm_mat, sol, name

    def __fetch_online(self):
        """
        Fetch from online QAPLIB data
        """
        for name in self.data_list:
            dat_content = urllib.request.urlopen(cfg.QAPLIB.ONLINE_REPO + 'data.d/{}.dat'.format(name)).read()
            sln_content = urllib.request.urlopen(cfg.QAPLIB.ONLINE_REPO + 'soln.d/{}.sln'.format(name)).read()

            dat_file = (self.qap_path / (name + '.dat')).open('wb')
            dat_file.write(dat_content)
            sln_file = (self.qap_path / (name + '.sln')).open('wb')
            sln_file.write(sln_content)

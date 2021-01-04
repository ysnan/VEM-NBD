import argparse
class get_test_config():
    def __init__(self):
        # Parse from command line
        self.parser = argparse.ArgumentParser(description='Non blind Deconvolution')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')

        # Training Parameters
        self.parser.add_argument('--layers', type=int, default=4, help='net layers')
        self.parser.add_argument('--deep', type=int, default=17, help='one module deep')
        self.parser.add_argument('--save_img', default=False, help='save images into file')
        self.parser.parse_args(namespace=self)


        self.beta = [0.005, 1, 1, 1, 1]

        # Result saving locations
        self.dataset_name = ['Set12'] # 'Sun', 'Levin'
        self.test_sigma = [2.55, 5.1, 7.65, 10.2, 12.75]
        self.test_ckp_dir = './pretrained_model/vem'
        self.test_bl_dir = {}
        self.test_sp_dir = {}

        self.ker_dir = './data/Set12/kernels/Levin09.mat'

        for dset in self.dataset_name:
            self.test_sp_dir[dset] = './data/{}/sharp/'.format(dset)
            for sigma in self.test_sigma:
                self.test_bl_dir[dset + '_' + str(sigma)]  = './data/'+ dset +'/sigma_' + str(sigma) + '_ker_levin_taper/BlurryNoiseDset/'

        self.test_save_dir = 'deblurred_results/'

        ## For Poisson Images
        # self.dataset_name = ['Pois_Set12']
        # self.test_sigma = [51,128,255,512,1024]
        # for dset in self.dataset_name:
        #     self.test_sp_dir[dset] = './data/Set12/sharp/'
        #     for sigma in self.test_sigma:
        #         self.test_bl_dir[dset + '_' + str(sigma)]  = './data/'+ dset +'/pois_' + str(sigma) + \
        #                                       '_ker_levin_taper/BlurryNoiseDset/'






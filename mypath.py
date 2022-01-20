class Path(object):
    @staticmethod
    def db_dir(database):
        if database == '360':
            # folder that contains class labels
            #root_dir = '/Path/to/UCF-101'
            root_dir = 'data/360-label01'

            # Save preprocess data into output_dir
            #output_dir = '/path/to/VAR/ucf101'
            output_dir = 'data/360-out'

            return root_dir, output_dir
        elif database == '360test':
            # folder that contains class labels
            root_dir = 'data/360-label01test'

            output_dir = 'data/360-out'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        # return '/path/to/Models/c3d-pretrained.pth'
        return 'models/ucf101-caffe.pth'
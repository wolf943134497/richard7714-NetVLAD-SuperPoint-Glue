import argparse

from modules.get_datamat import get_datamat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ref_path'
    )
    parser.add_argument(
        '--ref', type=str,
        default='/home/ma/Downloads/NaverLabs_Dataset/HyundaiDepartmentStore_mapping/1F/release/mapping/sensors'
                '/records_data/2019-04-16_14-35-00/images/'
    )
    parser.add_argument(
        '--valid', type=str,
        default='/home/ma/Downloads/NaverLabs_Dataset/HyundaiDepartmentStore_validation/1F/release/validation'
                '/sensors/records_data/2019-08-21_12-10-13/galaxy/'
    )

    opt = parser.parse_args()
    get_datamat(opt.ref,opt.valid)
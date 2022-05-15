from modules.Netvlad import NetVLAD
from modules.dataset_backup import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import argparse
import time

from torchvision.models import resnet18, vgg16

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='hitech_localization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ref', type=str, default='datasets/hitech/ref',
        help='location of ref image'
    )
    parser.add_argument(
        '--pose', type=str, default='pose.txt',
        help='txt file that has pose of picture'
    )
    parser.add_argument(
        '--query', type=str, default='datasets/hitech/query',
        help='location of query image'
    )
    parser.add_argument(
        '--thres', type=int, default=2,
        help='threshold for NetVLAD training'
    )
    opt = parser.parse_args()

    start = time.time()
    encoder = vgg16(pretrained=True)  # pretrained
    layers = list(encoder.features.children())[:-2]

    for l in layers[:-5]:
        for p in l.parameters():
            p.requires_grad = False

    model = nn.Module()

    encoder = nn.Sequential(*layers)
    model.add_module('encoder', encoder)

    dim = list(encoder.parameters())[-1].shape[0]  # last channels (512)

    # Define model for embedding
    net_vlad = NetVLAD(num_clusters=16, dim=dim)
    model.add_module('pool', net_vlad)

    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    model = model.cuda()

    from tqdm import tqdm

    train_feature_list = list()

    cluster_dataset = Dataset(condition="cluster")
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=1, shuffle=False, num_workers=0)

    from os import path
    if not path.exists('feature_savez.npz'):
    # 파일 존재 여부에 따라 작동
        with torch.no_grad():
            for batch_idx, train_image in tqdm(enumerate(cluster_loader)):
                output_train = model.encoder(train_image.cuda())
                output_train = model.pool(output_train)
                train_feature_list.append(output_train.squeeze().detach().cpu().numpy())
        train_feature_list_save = np.array(train_feature_list)
        np.savez('feature_savez',train_feature_list_save)
    else :
        loaded = np.load('feature_savez.npz')
        for i in loaded.keys():g
            train_feature_list = np.concatenate([loaded[i]])
            print(train_feature_list)
        print(train_feature_list.shape)

    import faiss
    import glob
    import torchvision.transforms as transforms
    from PIL import Image
    import pickle

    data_dict = {}

    faiss_index = faiss.IndexFlatL2(train_feature_list.shape[1])
    faiss_index.add(train_feature_list)

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    from pynput import keyboard

    key_on = 1


    def on_press(key):
        global key_on
        if str(key) == "'l'":
            key_on = 1


    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    # print("Ready\nPress any key to localize")

    opt_query = opt.query
    opt_ref = opt.ref
    ref_list = list(glob.glob(opt_ref + "/*.jpg"))
    while True:
        if key_on == 1:

            gb = glob.glob(opt_query + "/*.jpg")
            q_path = sorted(gb)

            for i in range(0, len(gb)):
                q_img = Image.open(q_path[i])
                q_img = q_img.resize((640, 480))
                q_img = input_transform(q_img)
                output_test = model.encoder(q_img.unsqueeze(dim=0).cuda())
                output_test = model.pool(output_test)
                query_feature = output_test.squeeze().detach().cpu().numpy()
                _, predictions = faiss_index.search(query_feature.reshape(1, -1), len(ref_list)//100)

                q_name = q_path[i].split('query/')
                temp_list = []

                for idx in predictions[0]:
                    ref_path = str(ref_list[idx]).split("hitech/")
                    ref_name = ref_path[1].split('ref/')
                    temp_list.append(ref_name[1])

                data_dict[str(q_name[1])] = temp_list

            with open('data_dict.pkl', 'wb') as f:
                pickle.dump(data_dict, f)

            # exec(open("models/SIFT_feature_matching.py").read())
            # key_on = 0
            # end = time.time()
            # f = open("outputs/matched_sift/match.txt",'a')
            # f.write("\n\ntime : %s\n" %(end-start))
            # f.close()

            # exec(open("models/SURF_feature_matching.py").read())
            # key_on = 0
            # end = time.time()
            # f = open("outputs/matched_surf/match.txt",'a')
            # f.write("\n\ntime : %s\n" %(end-start))
            # f.close()

            # exec(open("models/ORB_feature_matching.py").read())
            # key_on = 0
            # end = time.time()
            # f = open("outputs/matched_orb/match.txt",'a')
            # f.write("\n\ntime : %s\n" %(end-start))
            # f.close()

            exec(open("hitech.py").read())
            key_on = 0
            end = time.time()
            # f = open("outputs/matched/match.txt", 'a')
            # f.write("\ntime : %s \n" % (end - start))
            # f.close()

            break
            # print("\nPress any key to localize")

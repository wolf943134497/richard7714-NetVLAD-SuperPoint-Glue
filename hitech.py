from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch

import pickle
import glob
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='datasets/hitech/ref',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default='outputs/hitech',
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--display', action='store_true',
        help='Do display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)

    with open('data_dict.pkl', 'rb') as f:
        mydict = pickle.load(f)

    with open('pose_ref.pkl', 'rb') as r:
        pose_ref = pickle.load(r)

    with open('pose_valid.pkl', 'rb') as q:
        pose_q = pickle.load(q)

    output_image = []  # match개수가 가장 많은 하나의 이미지를 기록할 list
    q_name = []
    gb = sorted(glob.glob('datasets/hitech/query/*.jpg'))
    for i in range(0, len(gb)):
        q_name.append(gb[i].split('query/')[1])

    index = list(mydict.keys())

    file_generated = False

    for idx in index:
        impath = 'datasets/hitech/query/' + str(idx)
        query = vs.load_image(str(impath))

        frame_tensor = frame2tensor(query, device)
        last_data = matching.superpoint({'image': frame_tensor})
        last_data = {k + '0': last_data[k] for k in keys}
        last_data['image0'] = frame_tensor
        last_frame = query
        last_image_id = index.index(idx) + 1

        if opt.output_dir is not None:
            print('==> Will write outputs to {}'.format(opt.output_dir))
            Path(opt.output_dir).mkdir(exist_ok=True)

        # Create a window to display the demo.
        if opt.display:
            cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('SuperGlue matches', 640 * 2, 480)
        else:
            print('Skipping visualization, will not show a GUI.')

        timer = AverageTimer()

        match_num = []  # matching ratio 기록

        for it in mydict[impath.split('query/')[1]]:
            ref_path = 'datasets/hitech/ref/' + it
            frame = vs.load_image(ref_path)

            timer.update('data')
            stem0, stem1 = last_image_id, int(it.split(".jpg")[0])

            frame_tensor = frame2tensor(frame, device)
            pred = matching({**last_data, 'image1': frame_tensor})
            kpts0 = last_data['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            timer.update('forward')

            valid = matches > -1
            mkpts0 = kpts0[valid]
            match_num.append(len(mkpts0) / len(kpts1))

            mkpts1 = kpts1[matches[valid]]
            color = cm.jet(confidence[valid])
            matching_ratio = len(mkpts0)/ len(kpts1)

            text = [
                'Only SuperPoint',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matching ratio: {}'.format(matching_ratio)
            ]
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                # 'Keypoint Threshold: {:.4f}'.format(k_thresh),
                # 'Match Threshold: {:.2f}'.format(m_thresh),
                # 'Image Pair: {:06}:{:06}'.format(stem0, stem1),
            ]
            out = make_matching_plot_fast(
                last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

            if opt.display:
                cv2.imshow('SuperGlue matches', out)
                key = chr(cv2.waitKey(1) & 0xFF)
                if key == 'q':
                    vs.cleanup()
                    print('Exiting (via q) demo_superglue.py')
                    break
                elif key == 'n':  # set the current frame as anchor
                    last_data = {k + '0': pred[k + '1'] for k in keys}
                    last_data['image0'] = frame_tensor
                    last_frame = frame
                    last_image_id = (vs.i - 1)
                elif key in ['e', 'r']:
                    # Increase/decrease keypoint threshold by 10% each keypress.
                    d = 0.1 * (-1 if key == 'e' else 1)
                    matching.superpoint.config['keypoint_threshold'] = min(max(
                        0.0001, matching.superpoint.config['keypoint_threshold'] * (1 + d)), 1)
                    print('\nChanged the keypoint threshold to {:.4f}'.format(
                        matching.superpoint.config['keypoint_threshold']))
                elif key in ['d', 'f']:
                    # Increase/decrease match threshold by 0.05 each keypress.
                    d = 0.05 * (-1 if key == 'd' else 1)
                    matching.superglue.config['match_threshold'] = min(max(
                        0.05, matching.superglue.config['match_threshold'] + d), .95)
                    print('\nChanged the match threshold to {:.2f}'.format(
                        matching.superglue.config['match_threshold']))
                elif key == 'k':
                    opt.show_keypoints = not opt.show_keypoints

            timer.update('viz')
            timer.print()

            if opt.output_dir is not None and matching_ratio > 0.2:
                # stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
                stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
                out_file = str(Path(opt.output_dir, stem + '.png'))
                print('\nWriting image to {}'.format(out_file))
                cv2.imwrite(out_file, out)

        cv2.destroyAllWindows()
        vs.cleanup()
        current_image = impath.split('query/')[1]
        l = mydict[current_image]
        output_image = l[match_num.index(max(match_num))]
        odom_of_ref = pose_ref[l[match_num.index(max(match_num))]]
        odom_of_q = pose_q[current_image]

        dist = np.linalg.norm(np.array(odom_of_ref) - np.array(odom_of_q))

        if not file_generated:
            f = open("outputs/matched/match.txt", 'w')
            f.write("Query Reference Distance Match_ratio\n")
            file_generated = True

        elif file_generated:
            f = open("outputs/matched/match.txt", 'a')

        f.write("%s %s " % (str(idx), output_image))
        f.write("%s " % dist)
        f.write("%s\n" % max(match_num))
        f.close()

        # print(odoms_ref[l[match_num.index(max(match_num))]])
        # print(odoms_q[current_image])

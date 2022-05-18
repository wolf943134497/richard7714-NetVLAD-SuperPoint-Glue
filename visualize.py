import pickle
import glob
import numpy as np

# Parameter
Threshold = 5

# pose 얻기
with open('pose_valid.pkl', 'rb') as _:
    pos_query = pickle.load(_)

with open('pose_ref.pkl', 'rb') as _:
    pos_ref = pickle.load(_)

gb = glob.glob('/home/ma/Downloads/SuperGlue_final/outputs/matched/*.txt')
for txt in gb:
    # localization 맞은 횟수
    correct = 0
    total = 0

    filename = txt.split('matched/')[1].split('.txt')[0]


    # match 결과
    with open(txt, 'rb') as _:
        data = _.readlines()

        # bytes -> str 타입으로 변경 : 'b 제거
        for i in range(len(data)):
            data[i] = data[i].decode('utf-8')

    # match 결과 및 threshold에 따른 dist T/F 저장
    query_len = len(data)
    ref_len = len(pos_ref)
    match = [[[] for __ in range(4)] for _ in range(len(data))]
    dist = [[] for _ in range(len(data))]

    # data를 추출하여 threshold에 따라 True,False 구분
    # 0 -> Query, 1 -> Ref, 2 -> dist, 3 -> match_ratio
    for i in range(1, query_len):
        match[i][:] = map(str, str(data[i]).split())
        if float(match[i][2]) <= Threshold:
            dist[i] = True
        else:
            dist[i] = False

    # plotting
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For line plot
    pos_q_list = [[] for _ in range(query_len)]
    pos_r_list = [[] for _ in range(query_len)]

    # Dictionary에서 각각의 좌표 획득
    for i in range(1, query_len):
        pos_q_list[i] = pos_query[match[i][0]]
        pos_r_list[i] = pos_ref[match[i][1]]

    # 선 그리기
    # 선을 하나씩 그려주어야 함
    # plot([x1,x2], [y1,y2], [z1,z2])
    # 1 <-> 2 연결하는 선 그리기
    # dist T/F에 따라 색상 구분
    for i in range(1, query_len):
        if dist[i]:
            ax.plot([pos_q_list[i][0], pos_r_list[i][0]], [pos_q_list[i][1], pos_r_list[i][1]], [0, 5], c='g')
            correct += 1
        else:
            ax.plot([pos_q_list[i][0], pos_r_list[i][0]], [pos_q_list[i][1], pos_r_list[i][1]], [0, 5], c='r')
        total += 1

    # 모든 점 trace 그리기
    ax.plot([pos_q_list[i][0] for i in range(1, query_len)], [pos_q_list[j][1] for j in range(1, query_len)], [0], marker = 'o',
            c='k')
    ax.plot([pos_ref[i][0] for i in pos_ref.keys()], [pos_ref[j][1] for j in pos_ref.keys()], [5], c='k')

    plt.axis('off')

    title = txt.split('matched/')
    ax.set_title(title[1].split('.txt')[0] + '\n' + 'Accuracy : ' + str(round(correct / total, 2) * 100) + '%')

    plt.savefig(filename)

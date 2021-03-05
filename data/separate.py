import os
import shutil

for i in range(10):
    path = 'lfw-set' + str(i)
    if not os.path.exists(path):
        os.makedirs(path)
    if i == 9:
        image_pair = 'lfw_funneled/pairs_10.txt'
    else:
        image_pair = 'lfw_funneled/pairs_0' + str(i + 1) + '.txt'
    with open(image_pair, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.strip() != '':
                old_path = 'lfw_funneled/' + line
                arr = line.split('/')
                if not os.path.exists('lfw-set' + str(i) + '/' + arr[0]):
                    os.makedirs('lfw-set' + str(i) + '/' + arr[0])
                new_path = 'lfw-set' + str(i) + '/' + line
                shutil.copy(old_path, new_path)

os.makedirs('lfw_train')
os.makedirs('lfw_test')

for i in range(10):
    path = 'lfw-set' + str(i)
    if not os.path.exists(path):
        os.makedirs(path)
    if i == 9:
        image_pair = 'lfw_funneled/pairs_10.txt'
    else:
        image_pair = 'lfw_funneled/pairs_0' + str(i + 1) + '.txt'
    with open(image_pair, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if line.strip() != '':
                old_path = 'lfw_funneled/' + line
                arr = line.split('/')
                current = ''
                if i == 9:
                    current = 'lfw_test/'
                else:
                    current = 'lfw_train/'
                if not os.path.exists(current + arr[0]):
                    os.makedirs(current + arr[0])
                new_path = current + line
                shutil.copy(old_path, new_path)

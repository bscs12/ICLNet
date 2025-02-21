import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import argparse


def generate_txt(parser):
    args = parser.parse_args()

    mode = args.mode
    input_path = args.input_path
    output_path = args.output_path

    names = []
    for name in os.listdir(input_path):
        name = os.path.splitext(name)[0]
        names.append(name)

    if mode == 'train':
        file = open(output_path+'/train.txt', 'w+')
        for i in names:
            file.write(i+'\n')
        file.close()
    elif mode == 'test':
        file = open(output_path+'/test.txt', 'w+')
        for i in names:
            file.write(i+'\n')
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train | test
    parser.add_argument("--mode", default='train')
    # 'datasets/AMUBUS/Train/Image' | 'datasets/AMUBUS/Test/Image'
    parser.add_argument("--input_path", default='datasets/AMUBUS/Train/Image')
    # 'datasets/AMUBUS/Train' | 'datasets/AMUBUS/Test'
    parser.add_argument("--output_path", default='datasets/AMUBUS/Train')
    generate_txt(parser)

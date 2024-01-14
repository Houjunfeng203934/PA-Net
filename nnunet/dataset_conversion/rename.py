import os

def main(path):
    filename_list = os.listdir(path)
    print(filename_list)


    a = 0
    for i in filename_list:
        used_name = path + filename_list[a]
        new_name = path + "volume-" + i[8:-10] + ".nii"
        os.rename(used_name, new_name)
        print(used_name, new_name)
        a += 1
if __name__ == '__main__':
    path = "/home/hjf/dataset/mul-ph/2/"
    main(path)


import argparse

def calc_time(sec):
    h = int(sec / 3600)
    m = int((sec % 3600) / 60)
    s = int(sec % 60)
    time = "{0}:{1}:{2}".format(h, m, s)

    return time 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--time', type=int, default=10)
    args = parser.parse_args()
    return args

def sum_time(filepath):
    sum_sec = 0.0

    with open(filepath, "r") as f:
        for line in f.readlines():
            end = float(line.split(" ")[3])
            start = float(line.split(" ")[2])
            sum_sec += end - start
    
    time = calc_time(sum_sec)
    print(time)

def specified_time(filepath, sp_time):
    sum_sec = 0.0
    row = 0

    with open(filepath, "r") as f:
        for line in f.readlines():
            end = float(line.split(" ")[3])
            start = float(line.split(" ")[2])
            sum_sec += end - start
            if sum_sec > sp_time:
                break
            row += 1
    
    print(row)

def main():
    args = get_args()
    filepath = args.dir + "/segments"

    sum_time(filepath)
    specified_time(filepath, args.time)

if __name__ == "__main__":
    main()
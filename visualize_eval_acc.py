import matplotlib.pyplot as plt

log_file_dict = {#"train_results/2018_10_21_21_55/log.txt": "misuse bn during validation",
                 "train_results/2018_10_22_01_16/log.txt": "basic pointnet",
                 # "train_results/2018_10_22_13_32/log.txt": "not use bn before global feature extraction",
                 #"train_results/2018_10_22_16_53/log.txt": "not use bn before global feature extraction",
                    "train_results/2018_10_22_19_32/log.txt": "not use bn before global feature extraction;dependency issue fix",
                    "train_results/2018_10_23_00_34/log.txt": "ln",
                    "train_results/2018_10_23_09_19/log.txt": "bn",}

plt.figure()
for log_dir in log_file_dict.keys():
    with open(log_dir, "r") as f:
        eval_acc = [float(i.rstrip().split(" ")[-1]) for i in f.readlines() if "eval overall" in i]
    plt.plot(eval_acc, label=log_file_dict[log_dir])
plt.ylabel("validation acc")
plt.xlabel("epoch")
plt.legend()
plt.show()

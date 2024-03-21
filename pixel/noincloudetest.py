import os

save_folder = ''
test_folder = ''

with open(os.path.join(save_folder, "pixcl_ca_mixup_sar_91.list"), "r") as fp:
    lines1 = fp.readlines()

# with open(os.path.join(save_folder, "lq_wotest.list"), "r") as fp:
#     lines2 = fp.readlines()

testnames = os.listdir(test_folder)

with open(os.path.join(save_folder, "pixcl_ca_mixup_sar_91_wotest.list"), "w") as fp:
    for line in lines1:
        name = os.path.basename(line.rstrip('\n'))
        if testnames.count(name) == 0:
            fp.write(line)
        else:
            print(name)

# with open(os.path.join(save_folder, "lq_wotest_2.list"), "w") as fp:
#     for line in lines2:
#         name = os.path.basename(line.rstrip('\n'))
#         if testnames.count(name) == 0:
#             fp.write(line)
#         else:
#             print(name)

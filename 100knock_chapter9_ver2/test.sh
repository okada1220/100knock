#!/bin/sh
dict_filepath="../100knock_chapter6/train.txt"
train_filepath="../100knock_chapter6/train.txt"
test_filepath="../100knock_chapter6/test.txt"
weight_filepath="../100knock_chapter7/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
case "$1" in
 "80") /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE 100knock_chapter9_ver2_80.py ${dict_filepath}
 ;;
 "81") /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE 100knock_chapter9_ver2_81.py ${dict_filepath}
 ;;
 "82") /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE 100knock_chapter9_ver2_82.py ${dict_filepath} ${train_filepath} ${test_filepath}
 ;;
 "83")
 a=1
 filename="100knock_chapter9_ver2_83($2).py"
 while [ $a -lt 16 ]
 do
   echo "バッチサイズ：$a"
   /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE ${filename} ${dict_filepath} ${train_filepath} $a
   a=`expr $a * 2`
  done
  ;;
 "84") /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE 100knock_chapter9_ver2_84.py ${dict_filepath} ${train_filepath} ${weight_filepath} 1
 ;;
 "85") /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE 100knock_chapter9_ver2_85.py ${dict_filepath} ${train_filepath} 1 2
 ;;
 "86") /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE 100knock_chapter9_ver2_86.py ${dict_filepath}
 ;;
 "87") /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE 100knock_chapter9_ver2_87.py ${dict_filepath} ${train_filepath} ${test_filepath}
 ;;
 "88") /mnt/c/Users/naoki/Anaconda3/envs/100knock/python.EXE 100knock_chapter9_ver2_88.py ${dict_filepath} ../100knock_chapter6/minimini-train.txt ../100knock_chapter6/minimini-test.txt
 ;;
esac
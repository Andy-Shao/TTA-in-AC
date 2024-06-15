#sh pre_train.sh
#sleep 30

sh STDA.sh
sleep 30

sh bridge_MTDA.sh
sleep 30
sh MTDA.sh

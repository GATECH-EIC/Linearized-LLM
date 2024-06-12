export SETTING="allaug-9"
#############
#alllocal -> local
#allglobal -> grouped 
#allboth -> local + grouped
#allboth-alpha -> local + alpha * grouped
#allaug-9 -> convlution augmentation with kernel size of 9
textclassification=(  "sst2"  )

#"sst2"  "mrpc"  "stsb"  "qqp"  "mnli"  "qnli"  "wnli"
mkdir -p ./tmp/$SETTING
for task in ${textclassification[*]}; do
    echo "run task $task"
    TASK=$task bash t5-settings/$SETTING.sh   2>&1 | tee -a ./tmp/$SETTING/$task.log
done

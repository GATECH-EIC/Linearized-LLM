export SETTING="aug"
#SETTING=local -> local 
#SETTING=global -> grouped
#SETTING=aug -> augmented LA
textclassification=("sst2" )
 
mkdir -p ./xtmp/$SETTING
for task in ${textclassification[*]}; do
    echo "run task $task"
    TASK=$task bash gpt2-settings/$SETTING.sh  2>&1 | tee -a ./xtmp/$SETTING/$task.log
done

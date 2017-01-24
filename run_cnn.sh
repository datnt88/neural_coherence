
### <- set paths - >


data="./final_data/"

CNN_SCR="cnn_ext_coherence.py"
EXP_DIR="saved_exp/"
MODEL_DIR="saved_models/"

mkdir -p $MODEL_DIR
mkdir -p $EXP_DIR

###<- Set general DNN settings ->
dr_ratios=(0.5 0.2) #dropout_ratio
mb_sizes=(32 16) #minibatch-size

### <- set CNN settings ->
nb_filters=(150 200 250) #no of feature map
w_sizes=(6 7 8 5)
pool_lengths=(6 5 4)



log=$1 
echo "Training...!" > $log

for ratio in ${dr_ratios[@]}; do
	for nb_filter in ${nb_filters[@]}; do
		for w_size in ${w_sizes[@]}; do
			for pool_len in ${pool_lengths[@]}; do
				for mb in ${mb_sizes[@]}; do

					echo "INFORMATION: dropout_ratio=$ratio filter-nb=$nb_filter w_size=$w_size pool_len=$pool_len minibatch-size=$mb" >> $log;
					echo "----------------------------------------------------------------------" >> $log;

					THEANO_FLAGS=device=gpu,floatX=float32 python $CNN_SCR --data-dir=$data --model-dir=$MODEL_DIR \
							--dropout_ratio=$ratio --minibatch-size=$mb\
							--nb_filter=$nb_filter --w_size=$w_size --pool_length=$pool_len >>$log
					wait

					echo "----------------------------------------------------------------------" >> $log;
				done
			done 
		done	
	done 
done

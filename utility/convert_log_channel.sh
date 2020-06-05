#sh
DATA_DIR="./data/channel/"

MODEL="alexnet"
for NODE in 4 8 16 32
do
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 4
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 5
done

MODEL="vgg"
for NODE in 4 8 16 32 64
do
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 12
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 13
done

MODEL="resnet"
for NODE in 4 8 16 32 64
do
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 52
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 53
done
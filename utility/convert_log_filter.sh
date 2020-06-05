#sh
# DATA_DIR="./data/filter/"

# MODEL="alexnet"
# for NODE in 4 8 16 32
# do
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 5
	
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 5
# done

# # MODEL="vgg"
# # for NODE in 4 8 16 32 64
# # do
	# # FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	# # python convert_log.py  -infile ${FILENAME} -maxline 13
	
	# # FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	# # python convert_log.py  -infile ${FILENAME} -maxline 13
# # done

# MODEL="resnet"
# for NODE in 4 8 16 32 64
# do
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 53
	
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 53
# done


# #### FOR FILTER + DATA:
DATA_DIR="./data/df/results"

MODEL="alexnet"
for NODE in 8 16 32 64 128 256 512 1024 
do
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/*.out"
	#rm ${FILENAME}
	let GROUP=$NODE/4
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"	
	python convert_log.py  -infile "${FILENAME}.out" -maxline 5
	rm "${FILENAME}.out"
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	python convert_log.py  -infile "${FILENAME}.out" -maxline 5
	rm "${FILENAME}.out"

	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/gradient_allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 4 --m "MAX"
done

MODEL="vgg"
for NODE in 8 16 32 64 128 256 512 1024 
do
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/*.out"
	#rm ${FILENAME}
	let GROUP=$NODE/4
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"	
	python convert_log.py  -infile "${FILENAME}.out" -maxline 13
	rm "${FILENAME}.out"
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	python convert_log.py  -infile "${FILENAME}.out" -maxline 13
	rm "${FILENAME}.out"

	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/gradient_allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 4 --m "MAX"
done


#python convert_log.py  -infile $"./data/df//resnet/8/allgather_times.log" -maxline 2 --m "MAX"	

MODEL="resnet"
for NODE in 8 16 32 64 128 256 512 1024 
do
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/*.out"
	#rm ${FILENAME}
	let GROUP=$NODE/4
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"	
	python convert_log.py  -infile "${FILENAME}.out" -maxline 53
	rm "${FILENAME}.out"
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/allgather_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	python convert_log.py  -infile "${FILENAME}.out" -maxline 53
	rm "${FILENAME}.out"

	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/gradient_allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 4 --m "MAX"
done

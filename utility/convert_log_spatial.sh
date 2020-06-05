#sh
# DATA_DIR="./data/cosmoflow/results"

# for NODE in 8 16 32 64 128 256 512
# do
	# let GROUP=$NODE/4
	# let HG=2
	# # FILENAME="${DATA_DIR}/${MODEL}/${NODE}/backward_halo_exchange_time_file.log"
	# # python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	# # #python convert_log.py  -infile "${FILENAME}.out" -maxline 2
	
	# # FILENAME="${DATA_DIR}/${MODEL}/${NODE}/forward_halo_exchange_time.log"
	# # python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	# # #python convert_log.py  -infile "${FILENAME}.out" -maxline 2
	
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/forward_spatial_allgather.log"
	# python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	# #python convert_log.py  -infile "${FILENAME}.out" -maxline 2
	
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/backward_spatial_allgather.log"
	# python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	# #python convert_log.py  -infile "${FILENAME}.out" -maxline 2
	
	# # FILENAME="${DATA_DIR}/${MODEL}/${NODE}/gradient_allreduce_times.log"
	# # python convert_log.py  -infile ${FILENAME} -maxline 2 --m "MIN"
	
# done

# DATA_DIR="./data/spatial/"
# MODEL="alexnet"
# for NODE in 4 
# do
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/backward_halo_exchange_time_file.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 4
	
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/forward_halo_exchange_time.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 5
# done

# MODEL="vgg"
# for NODE in 4 
# do
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/backward_halo_exchange_time_file.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 13
	
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/forward_halo_exchange_time.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 13
# done

# MODEL="resnet"
# for NODE in 4 
# do
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/backward_halo_exchange_time_file.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 53
	
	# FILENAME="${DATA_DIR}/${MODEL}/${NODE}/forward_halo_exchange_time.log"
	# python convert_log.py  -infile ${FILENAME} -maxline 1
# done

DATA_DIR="./data/ds/results/"

MODEL="vgg"
for NODE in 8 16 32 64 128 256 512 1024
do
	let GROUP=$NODE/4
	# # FILENAME="${DATA_DIR}/${MODEL}/${NODE}/backward_halo_exchange_time_file.log"
	# # python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	# # #python convert_log.py  -infile "${FILENAME}.out" -maxline 2

	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/backward_halo_exchange_time_file.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	python convert_log.py  -infile "${FILENAME}.out"  -maxline 53
	rm *.out
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/forward_halo_exchange_time.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	python convert_log.py  -infile "${FILENAME}.out" -maxline 1
	rm *.out
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/backward_spatial_allgather.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/forward_spatial_allgather.log"
	python convert_log.py  -infile ${FILENAME} -maxline "${GROUP}" --m "MAX"
	
	FILENAME="${DATA_DIR}/${MODEL}/${NODE}/gradient_allreduce_times.log"
	python convert_log.py  -infile ${FILENAME} -maxline 2 --m "MIN"
done
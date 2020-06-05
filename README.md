# paraDL-analysis
The tools for analysis the computation, communication and required memory for many types of distributed deep learning on HPC system.
# Required:
python version 2.x (code under Python 2.7.5)


# Usage
sage: parallel_analysis.py [-h] [-net NET] [-plat PLAT] [-goal GOAL]
                            [--lc LC] [--cmaxB CMAXB] [--cmaxp CMAXP]
                            [--cBon CBON] [--paratype PARATYPE]
                            [--debug DEBUG]

optional arguments:
  -h, --help           show this help message and exit
  -net NET             filename of the dataset specification and the model
                       specification(*.net)
  -plat PLAT           filename of the computer system specification (*.plat)
  -goal GOAL           Goal of the analysis 1.Performance 2. Memory 3. Cost
  --lc LC              layer configuration: filename contain applied layers.
                       Other layer is in sequential mode
  --cmaxB CMAXB        Constrain: Maximum mini-Batch size
  --cmaxp CMAXP        Constrain: Maximum number of node
  --cBon CBON          Constrain: Micro-batch size per node
  --paratype PARATYPE  parallelism type a: all, o: sequential in one PE, s:
                       spatial, p:pipeline, f:filter, c:channel,d:data.
                       Multiple type can be seperated by ",". Other than that
                       it is hybrid. For example, ds refers to hybrid of data
                       + spatial but d|s refers to analysis of data and
                       spatial
  --debug DEBUG        Debug mode. any character mean yes

# Examples 
 python parallel_analysis.py -net ./RESNET50_ImageNet_profall.net_sc  -plat ABCI.plat -goal 1 --cmaxp 8 --cmaxB 262144 --cBon=256 --paratype ds  --debug y --lc spatial_resnet.lc
or

 python parallel_analysis.py -net ./RESNET50_ImageNet_profall.net_sc  -plat ABCI.plat -goal 1 --cmaxp 1024 --cmaxB 262144 --cBon=64 --paratype d,f,c,ds,df  --debug y 

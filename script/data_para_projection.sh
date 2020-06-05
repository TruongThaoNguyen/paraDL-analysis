python parallel_analysis.py -net ./ALEXNET_ImageNet_profall.net_sc  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 2000000 --cBon=512 --paratype o,d --debug y >> data_alexnet_scaling_proj.log 
python parallel_analysis.py -net ./VGG_ImageNet_profall.net_sc  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=32 --paratype o,d --debug y >> data_vgg_scaling_proj.log 
python parallel_analysis.py -net ./RESNET50_ImageNet_profall.net_sc  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=64 --paratype o,d --debug y >> data_resnet_scaling_proj.log 

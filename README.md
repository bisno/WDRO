
# WDRO
The implementation on Towards Scalable and Fast Distributionally Robust Optimization for Data-Driven Deep Learning.

****

Dev requirement:

```
tensorflow-gpu-1.13.1
cvxopt-1.2.5
numpy-1.15.0
pandas-0.25.2
```

#### Toydata:
```
python WDRO_for_toydata.py
```

#### Implemtation:  
1, DNN+WDRO:


```
python 10-nonlinearized-dual.py --num_of_group 4 --num_minor 2 --train_subgroup_batch 100
``` 
Here,  
--num_of_group : total subgroup number $N$ ;    
--num_minor    : number of minor-class subgroup $c$ ;      
--train_subgroup_batch : number of data in each subgroup $s_j$ ;   



2, ResNet+WDRO:

```
python nonlinearized-dual-ResNet.py --num_of_group 4 --num_minor 2 --train_subgroup_batch 100
```

The label now is "eyeglasses" in CelebA dataset, which can be changed in data_process.py. 

# PNRL
This repository provides reference codes of PNRL as described in the paper:

>**Predictive Network Representation Learning for Link Prediction.**
>**Zhitao Wang, Chengyao Chen and Wenjie Li.**
>**SIGIR, 2017.**  (https://dl.acm.org/citation.cfm?id=3080692)

## Environment
The code is only tested in the environment with following setting:  
&emsp;1. python 2.7  
&emsp;2. keras (1.1.0) (Using Theano Backend with Theano Version 0.9.0)  
&emsp;3. numpy (1.11.1)  
&emsp;4. scikit-learn (0.16.1)  
&emsp;5. networkx (1.10)  

## Dataset
All experimental datasets are public, you can find in the following links:
* **Facebook**: J. McAuley and J. Leskovec. Learning to Discover Social Circles in Ego Networks. NIPS, 2012. https://snap.stanford.edu/data/egonets-Facebook.html
* **Email**: R. Guimera, L. Danon, A. Diaz-Guilera, F. Giralt and A. Arenas, Physical Review E , vol. 68, 065103(R), (2003). http://deim.urv.cat/~alexandre.arenas/data/welcome.htm
* **U.S. Power Grid**: D. J. Watts and S. H. Strogatz, Nature 393, 440-442 (1998). http://www-personal.umich.edu/~mejn/netdata/
* **Condensed Matter Collaborations**: M. E. J. Newman, The structure of scientific collaboration networks, Proc. Natl. Acad. Sci. USA 98, 404-409 (2001). http://www-personal.umich.edu/~mejn/netdata/

## Data Format
Training, validation and testing graphs should be the format of edge lists (undirected):
>node1_id_int node2_id_int <weight_float, optional>
>1 2 1.0
>......

## Citing
    @inproceedings{Wang2017PNR,
    author = {Wang, Zhitao and Chen, Chengyao and Li, Wenjie},
    title = {Predictive Network Representation Learning for Link Prediction},
     booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
     series = {SIGIR '17},
     year = {2017},
     isbn = {978-1-4503-5022-8},
     location = {Shinjuku, Tokyo, Japan},
     url = {http://doi.acm.org/10.1145/3077136.3080692}
    } 

  

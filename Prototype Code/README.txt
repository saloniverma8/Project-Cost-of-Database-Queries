These are the files, that we added to the benchmarking framework, which can be found here: https://github.com/graphdeeplearning/benchmarking-gnns/tree/master-dgl-0.5.2
The Files were added like this:

In /
Zinc_Graph_Reg_implementation.ipynb
LogicalPlan_and_PhyicalPlan_graph_regression.ipynb

In /data/
logical_plan.py
physical_plan.py
LogicalPlan folder with pkl and prepare notebooks
PhysicalPlan folder with pkl and prepare notebooks
holistic_cost_dgl_pickles_job_light with graph pkls and label csvs
holistic_cost_dgl_pickles_job_light_physical with graph pkls and label csvs
holistic_cost_dgl_pickles_job_synthetic with graph pkls and label csvs
holistic_cost_dgl_pickles_job_synthetic_physical with graph pkls and label csvs
data.py updated

In /nets/molecules_graph_regression/
gcn_net.py updated for IMDb dataset
gated_gcn_net.py updated for IMDb dataset

The notebooks were for the training of the ZINC and IMDb Dataset. The graph pkls and label csvs are the data for the training.
The data.py, logical_plan.py and physical_plan.py files with the LogicalPlan and PhysicalPlan folders are for loading the data of the pkls and csvs.
For getting the dataset pkls, which are in the LogicalPlan and PhysicalPlan folder, the Notebooks in there had to be run. The prepare_LogicalPlanBigGraphs, -Undirected and -Unnormalized Notebooks were for different training approaches, which gave no good results. The prepare_LogicalPlan prepares the Logical Cardinality Estimation.
The nets needs to be updated for the IMDb dataset, where the embedding for the using of floats is changed.
The results folder contains the normalized and unnormalized predictions, labels and absolute errors, for the, in the paper mentioned, configurations.
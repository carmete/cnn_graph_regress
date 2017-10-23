#!/bin/bash                                                               
#SBATCH --job-name=gcn_regress             # Job name                        
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec          
#SBATCH --partition=atom-gpu            # Mention partition-name. default 
#SBATCH --output=out.txt                # Output file.                    
#SBATCH --gres=gpu:1                    # N number of GPU devices.            
#SBATCH -w node14
#SBATCH --mail-type=ALL                 # Enable email                    
#SBATCH --mail-user=joyneel.misra@students.iiit.ac.in     # Where to send mail  
#SBATCH --mem-per-cpu=20000             # Enter memory, default is 100M. 

source activate tf
rm -rf /users/joyneel.misra/experiments/cnn_graph_regress/summaries
rm -rf /users/joyneel.misra/experiments/cnn_graph_regress/checkpoints
 
python /users/joyneel.misra/experiments/cnn_graph_regress/usage_classify.py

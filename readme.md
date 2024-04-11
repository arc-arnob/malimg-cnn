### Steps to reproduce the code.
1. Run: 
    ```bash
    python train.py value_k
    ```

2. value_k is a variable between 0 to 9 which represents one of the folds.
3. The indices for each stratified fold is already stored in a pickle file (cross_validation_indices.pkl).
4. These indices correspond to removal of all classes whose sample size is less than two. 


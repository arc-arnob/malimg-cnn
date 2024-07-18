# CNNs for Malware Categorisation

In this project, we employed a CNN architecture to classify MalImg malware binaries, obtaining 93% of mean accuracy over 20 classes. Furthermore, we classified over 100GB of real malicious software from the darknet, reaching 82% of mean accuracy over more than 500 classes. A detailed description of the project's aim, experiments, and results can be found in our blog post at https://oasis-soul-ee5.notion.site/Applying-CNNs-to-Malware-Classification-5273d5a57d3341d991ccffacb45237cf

### Steps to reproduce the code.
1. Run: 
    ```bash
    python train.py value_k
    ```

2. value_k is a variable between 0 to 9 which represents one of the folds.
3. The indices for each stratified fold is already stored in a pickle file (cross_validation_indices.pkl).
4. These indices correspond to removal of all classes whose sample size is less than two. 

